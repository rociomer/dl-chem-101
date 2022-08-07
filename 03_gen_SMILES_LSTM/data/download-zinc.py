from tdc.generation import MolGen
from pathlib import Path
import rdkit
from rdkit import Chem


def save_split(smi_file : Path, smi_list : list) -> None:
    """
    Saves input list of SMILES to the specified file path.
    """
    smi_writer = rdkit.Chem.rdmolfiles.SmilesWriter(str(smi_file))
    for smi in smi_list:
        try:
            mol = rdkit.Chem.MolFromSmiles(smi[0])
            if mol is not None:
                smi_writer.write(mol)
        except:  # likely TypeError or AttributeError e.g. "smi[0]" is "nan"
            continue
    smi_writer.close()

if __name__ == "__main__":
    print(f"*** Downloading ZINC dataset using the TDC ***")
    data      = MolGen(name = "ZINC")
    split     = data.get_split()

    zinc_path = Path("./data/zinc/")
    print(f"*** Splitting the data and saving the splits in '{zinc_path}' ***")
    zinc_path.mkdir(exist_ok=True)
    save_split(smi_file=zinc_path.joinpath("train.smi"), smi_list=split["train"].values)
    save_split(smi_file=zinc_path.joinpath("test.smi"), smi_list=split["test"].values)
    save_split(smi_file=zinc_path.joinpath("valid.smi"), smi_list=split["valid"].values)

    print("*** Done ***", flush=True)
