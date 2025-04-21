from dictionary_learning import ActivationBuffer
from dictionary_learning.dictionary import AutoEncoder, GatedAutoEncoder, AutoEncoderNew, RelaxedArchetypalAutoEncoder
from dictionary_learning.utils import zst_to_generator
from nnsight import LanguageModel
from dictionary_learning.training import trainSAE
from dictionary_learning.trainers.standard import StandardTrainer
from dictionary_learning.trainers.p_anneal import PAnnealTrainer
from dictionary_learning.trainers.gdm import GatedSAETrainer
import torch as t

DEVICE = 'cuda:0'

model = LanguageModel('EleutherAI/pythia-70m-deduped', device_map=DEVICE, dispatch=True)
submodule = model.gpt_neox.layers[1].mlp
activation_dim = 512

data = zst_to_generator('/home/aaron/data/00.jsonl.zst')
buffer = ActivationBuffer(
    data,
    model,
    submodule,
    d_submodule=activation_dim,
    n_ctxs=1e4,
    refresh_batch_size=512,
    out_batch_size=8192,
    io='out',
    device=DEVICE
)

trainSAE(
    buffer,
    trainer_configs = [
        {
            'trainer' : StandardTrainer,
            'dict_class' : RelaxedArchetypalAutoEncoder,
            'activation_dim' : 512,
            'dict_size' : 64*512,
            'l1_penalty' : 8e-1,
            'steps': 30000,
            'layer': 3,
            'lm_name': 'EleutherAI/pythia-70m-deduped',
            'device' : DEVICE,
        },
        {
            'trainer' : StandardTrainer,
            'dict_class' : RelaxedArchetypalAutoEncoder,
            'activation_dim' : 512,
            'dict_size' : 64*512,
            'l1_penalty' : 1,
            'steps': 30000,
            'layer': 3,
            'lm_name': 'EleutherAI/pythia-70m-deduped',
            'device' : DEVICE,
        },
    ],
    steps=30000,
    save_steps=(5000, 10000, 15000, 20000, 25000, 30000),
    save_dir = 'weights/{run}',
    log_steps=100,
)
