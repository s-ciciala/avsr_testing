import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from config import args
from models.video_net import VideoNet
from models.lrs2_char_lm import LRS2CharLM
from data.lrs3_dataset import LRS3Main
from data.utils import collate_fn
from utils.general import evaluate

def main():

    np.random.seed(args["SEED"])
    torch.manual_seed(args["SEED"])
    gpuAvailable = torch.cuda.is_available()
    device = torch.device("cuda" if gpuAvailable else "cpu")
    kwargs = {"num_workers":args["NUM_WORKERS"], "pin_memory":True} if gpuAvailable else {}
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    #declaring the test dataset and test dataloader
    videoParams = {"videoFPS":args["VIDEO_FPS"]}
    testData = LRS3Main("extended_test", args["DATA_DIRECTORY"], args["MAIN_REQ_INPUT_LENGTH"], args["CHAR_TO_INDEX"], args["STEP_SIZE"],
                        videoParams)
    testLoader = DataLoader(testData, batch_size=args["BATCH_SIZE"], collate_fn=collate_fn, shuffle=True, **kwargs)

    # args["TRAINED_MODEL_FILE"] = args["PRETRAINED_VIDEO_MODEL_FILE"]
    args["TRAINED_MODEL_FILE"] = args["TRAINED_VIDEO_MODEL_FILE"]

    if args["TRAINED_MODEL_FILE"] is not None:

        print("\nTrained Model File: %s" %(args["TRAINED_MODEL_FILE"]))

        #declaring the model, loss function and loading the trained model weights
        model = VideoNet(args["TX_NUM_FEATURES"], args["TX_ATTENTION_HEADS"], args["TX_NUM_LAYERS"], args["PE_MAX_LENGTH"],
                         args["TX_FEEDFORWARD_DIM"], args["TX_DROPOUT"], args["NUM_CLASSES"])
        saved_state_dict = torch.load(args["TRAINED_MODEL_FILE"], map_location=device)

        try:
            model_epoch = saved_state_dict["epoch"]
            model_state_dict = saved_state_dict["model_state_dict"]
            optimizer_state_dict = saved_state_dict["optimizer_state_dict"]
            model_loss = saved_state_dict["loss"]
            new_state_dict = {}
            for k, v in model_state_dict.items():
                name = k.replace('module.', '')  # remove the "module." prefix
                new_state_dict[name] = v
        except:
            new_state_dict = {}
            for k, v in saved_state_dict.items():
                name = k.replace('module.', '')  # remove the "module." prefix
                new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        model.to(device)
        loss_function = nn.CTCLoss(blank=0, zero_infinity=True)


        # #declaring the language model
        # lm = LRS2CharLM()
        # lm.load_state_dict(torch.load(args["TRAINED_LM_FILE"], map_location=device))
        # lm.to(device)
        if not args["USE_LM"]:
            lm = None


        print("\nTesting the trained model .... \n")

        beamSearchParams = {"beamWidth":args["BEAM_WIDTH"], "alpha":args["LM_WEIGHT_ALPHA"], "beta":args["LENGTH_PENALTY_BETA"],
                            "threshProb":args["THRESH_PROBABILITY"]}
        testParams = {"decodeScheme":args["TEST_DEMO_DECODING"], "beamSearchParams":beamSearchParams, "spaceIx":args["CHAR_TO_INDEX"][" "],
                      "eosIx":args["CHAR_TO_INDEX"]["<EOS>"], "lm":lm}

        #evaluating the model over the test set
        testLoss, testCER, testWER = evaluate(model, testLoader, loss_function, device, testParams)

        #printing the test set loss, CER and WER
        print("Test Loss: %.6f || Test CER: %.3f || Test WER: %.3f" %(testLoss, testCER, testWER))
        print("\nTesting Done.\n")


    else:
        print("Path to the trained model file not specified.\n")

    return



if __name__ == "__main__":
    main()