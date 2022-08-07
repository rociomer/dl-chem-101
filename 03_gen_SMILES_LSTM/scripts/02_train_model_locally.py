"""
Train a SMILES LSTM using ZINC data locally.

Model trains for 10 epochs using the default settings below (~5 mins on a GPU).
"""
import argparse
from pathlib import Path
from smiles_lstm.model.smiles_lstm import SmilesLSTM
from smiles_lstm.model.smiles_trainer import SmilesTrainer
from smiles_lstm.model.smiles_vocabulary import SMILESTokenizer, create_vocabulary
from smiles_lstm.utils import load
from smiles_lstm.utils.misc import suppress_warnings

# suppress minor warnings
suppress_warnings()

# define the argument parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 add_help=False)

# define parameters for the model
parser.add_argument("--output",
                    type=str,
                    default="./output/run_local/",
                    help="Specifies the path to which to write output to.")
args = parser.parse_args()


if __name__ == '__main__':
    print("*** Building a vocabulary from ZINC data ***", flush=True)
    zinc_path = Path("./data/zinc/")
    train     = load.smiles(path=(zinc_path.joinpath("train.smi")))
    test      = load.smiles(path=(zinc_path.joinpath("test.smi")))
    valid     = load.smiles(path=(zinc_path.joinpath("valid.smi")))
    
    dataset = train + test + valid
    
    test_tokenizer = SMILESTokenizer()
    test_vocab     = create_vocabulary(smiles_list=dataset,
                                       tokenizer=test_tokenizer,
                                       canonical=False)
    
    # trainer object expects data in a dictionary, so reorganizing it so
    SMILES_dict = {"train" : train, "valid" : valid, "test"  : test}
    
    # define network parameters
    network_parameters = {
        'num_layers'          : 3,
        'layer_size'          : 512,
        'cell_type'           : 'lstm',
        'embedding_layer_size': 512,
        'dropout'             : 0.2,
        'layer_normalization' : True,
    }
    
    print("*** Building the SMILES LSTM network and creating a Trainer object with it ***", flush=True)
    smiles_lstm = SmilesLSTM(vocabulary=test_vocab,
                             tokenizer=test_tokenizer,
                             network_params=network_parameters)
    
    trainer = SmilesTrainer(model=smiles_lstm,
                            input_smiles=SMILES_dict,
                            epochs=10,
                            shuffle=True,
                            batch_size=2048,
                            learning_rate=0.0001,
                            augment=3,
                            output_model_path=args.output,
                            start_epoch=0,
                            learning_rate_scheduler="StepLR",
                            gamma=0.9,
                            eval_num_samples=256,
                            eval_batch_size=256)
    
    print("*** Training the model ***", flush=True)
    trainer.run()
