import requests
from ase.db import connect 
from ase import Atoms
from ase import db
import os

import numpy as np
print("Start")
url = "https://sgdml.org/secure_proxy.php?file=data/npz/md17_benzene2017.npz"

headers = {
    "User-Agent": "Mozilla/5.0"
}

r = requests.get(url, headers=headers, allow_redirects=True)

print("Start")

print(r.status_code)
print(r.url)
print(r.headers.get("content-type"))

with open("md17_benzene2017.npz", "wb") as f:
    f.write(r.content)





data = np.load("./md17_benzene2017.npz")

numbers = data["z"]
atoms_list = []
property_list = []
for positions, energies, forces in zip(data["R"], data["E"], data["F"]):
    ats = Atoms(positions=positions, numbers=numbers)
    properties = {"energy": energies, "forces": forces}
    property_list.append(properties)
    atoms_list.append(ats)

class ASE_DB:
    def __init__(self,name, distance_unit, property_unit_dict):
       self.name = name
       self.distance_unit = distance_unit
       self.property_unit_dict = property_unit_dict
       self.db = connect(self.name)
    def writef(self, property_list, atoms_list):
        with self.db:
            for atoms, properties in zip(atoms_list, property_list):
                self.db.write(
                    atoms,
                    data={
                        "energy": properties["energy"],
                        "forces": properties["forces"]
                    }
                )

db_path =  './new_dataset.db'
if os.path.exists(db_path):
    os.remove(db_path)

db_path_lock =  './new_dataset.db.lock'
if os.path.exists(db_path_lock):
    os.remove(db_path_lock)

new_dataset = ASE_DB(
    "./new_dataset.db",
    distance_unit="Ang",
    property_unit_dict={"energy": "kcal/mol", "forces": "kcal/mol/Ang"},
)
new_dataset.writef(property_list, atoms_list)

print(new_dataset.name)