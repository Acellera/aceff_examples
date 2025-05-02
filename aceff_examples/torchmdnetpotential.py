# MIT License

# Copyright (c) 2025 Acellera info@acellera.com

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from openmm import NonbondedForce
from openmm.unit import elementary_charge
from openmmml.mlpotential import MLPotential, MLPotentialImpl, MLPotentialImplFactory
import openmmtorch
import torch as pt
from torchmdnet.models.model import load_model


class TorchMDNETForce(pt.nn.Module):

    def __init__(
        self,
        model_file,
        atomic_numbers,
        group_indices,
        total_charges,
        max_num_neighbors,
    ):
        super().__init__()
        assert len(group_indices) == len(total_charges)

        self.model = load_model(
            model_file,
            derivative=False,
            max_num_neighbors=max_num_neighbors,
            remove_ref_energy=False,
            static_shapes=True,
            check_errors=False,
        )
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.register_buffer(
            "all_atom_indices", pt.tensor(sum(group_indices, []), dtype=pt.long)
        )
        self.register_buffer(
            "atomic_numbers",
            pt.tensor(atomic_numbers, dtype=pt.long)[self.all_atom_indices],
        )
        batch = sum(
            [[i] * len(atom_indices) for i, atom_indices in enumerate(group_indices)],
            [],
        )
        self.register_buffer("batch", pt.tensor(batch, dtype=pt.long))
        self.register_buffer("total_charges", pt.tensor(total_charges, dtype=pt.long))

    def forward(self, positions):
        positions = (
            pt.index_select(positions, 0, self.all_atom_indices).to(pt.float32) * 10
        )  # nm --> A
        energies, *_ = self.model(
            self.atomic_numbers, positions, batch=self.batch, q=self.total_charges
        )
        return energies.sum() * 96.4915666370759  # eV -> kJ/mol


class TorchMDNETImpl(MLPotentialImpl):

    def __init__(
        self, name, model_file, group_indices, molecule_charges, max_num_neighbors, use_cuda_graphs
    ):
        self.name = name
        self.model_file = model_file
        self.group_indices = group_indices
        self.molecule_charges = molecule_charges
        self.max_num_neighbors = int(max_num_neighbors)
        self.use_cuda_graphs = bool(use_cuda_graphs)

    @staticmethod
    def _get_total_charge(system, atom_indices):
        for force in system.getForces():
            if isinstance(force, NonbondedForce):
                return round(
                    sum(
                        force.getParticleParameters(i)[0].value_in_unit(
                            elementary_charge
                        )
                        for i in atom_indices
                    )
                )

    def addForces(self, topology, system, all_atom_indices, force_group):
        
        atomic_numbers = [atom.element.atomic_number for atom in topology.atoms()]

        if self.molecule_charges is None:
            total_charges = [
                self._get_total_charge(system, atom_indices)
                for atom_indices in self.group_indices
            ]
        else:
            total_charges = self.molecule_charges

        if all_atom_indices is not None:
            assert sum(self.group_indices, []) == all_atom_indices

        force = TorchMDNETForce(
            self.model_file,
            atomic_numbers,
            self.group_indices,
            total_charges,
            self.max_num_neighbors,
        )
        force = openmmtorch.TorchForce(pt.jit.script(force))
        force.setProperty("useCUDAGraphs", "true" if self.use_cuda_graphs else "false")
        force.setForceGroup(force_group)
        system.addForce(force)


class TorchMDNETImplFactory(MLPotentialImplFactory):

    def createImpl(
        self, name, model_file, group_indices, molecule_charges=None, max_num_neighbors=40, use_cuda_graphs=True
    ):
        return TorchMDNETImpl(
            name, model_file, group_indices, molecule_charges, max_num_neighbors, use_cuda_graphs
        )


MLPotential.registerImplFactory("TorchMD-NET", TorchMDNETImplFactory())