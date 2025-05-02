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



import ase.calculators.calculator as ase_calc
import torch
from torchmdnet.models.model import load_model

ACEFF_ATOMIC_NUMBERS = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]

class AceFFCalculator(ase_calc.Calculator):

    implemented_properties = [
        "energy",
        "forces"
    ]

    def __init__(
        self,  model_file, device = 'cpu', **kwargs,
    ):

        ase_calc.Calculator.__init__(self, **kwargs)
        self.device=device
        self.model = load_model(
            model_file,
            derivative=True,
            max_num_neighbors=64,
            remove_ref_energy=False,
            static_shapes=False,
            check_errors=True,
        )
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.model.to(self.device)

    def calculate(
        self,
        atoms = None,
        properties = None,
        system_changes = ase_calc.all_changes,
    ):

        if not properties:
            properties = ["energy"]
        ase_calc.Calculator.calculate(self, atoms, properties, system_changes)

        numbers = atoms.numbers
        for number in numbers:
            assert(number in ACEFF_ATOMIC_NUMBERS)
        
        positions = atoms.positions
        total_charge = int(sum(atoms.get_initial_charges()))
        batch = [0 for _ in range(len(numbers))]

        numbers = torch.tensor(numbers, device=self.device, dtype=torch.long)
        positions = torch.tensor(positions, device=self.device, dtype=torch.float32)
        batch = torch.tensor(batch, device=self.device, dtype=torch.long)
        total_charge = torch.tensor([total_charge], device=self.device, dtype=torch.long)

        energies, forces = self.model(
            numbers, positions, batch=batch, q=total_charge
        )


        self.results["energy"] = energies.detach().cpu().item()
        self.results["forces"] = forces.detach().cpu().numpy()




