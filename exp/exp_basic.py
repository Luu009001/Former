import os
import torch
from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM,Proposed_ASBMGTUgd, CARD,CARD1,Proposed_ASBSEA,Proposed_SEA,Proposed_MGTUgd,Proposed_ASB,Proposed_SEACRMSA,Proposed_SEA_MGTUgd_ASB,Proposed_SEAMGTUgd,CARD,Proposed_robdecomp,Proposed_SEACRMSAgd,Proposed_SEAgdMGTUgd,Proposed_SEAgdMGTU,Proposed_SEA_MGTUgd_robdecomp


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'CARD': CARD,
            'CARD1': CARD1,
            'Proposed_SEA':Proposed_SEA,
            'Proposed_ASB':Proposed_ASB,
            'Proposed_MGTUgd':Proposed_MGTUgd,
            'Proposed_SEAMGTUgd':Proposed_SEAMGTUgd,
            'Proposed_ASBMGTUgd':Proposed_ASBMGTUgd,
            'Proposed_SEACRMSAgd':Proposed_SEACRMSAgd,
            'Proposed_SEAgdMGTUgd':Proposed_SEAgdMGTUgd,
            'Proposed_SEAgdMGTU':Proposed_SEAgdMGTU,
            'Proposed_robdecomp':Proposed_robdecomp,
            'Proposed_SEACRMSA':Proposed_SEACRMSA,
            'Proposed_SEA_MGTUgd_robdecomp':Proposed_SEA_MGTUgd_robdecomp,
            'Proposed_SEA_MGTUgd_ASB':Proposed_SEA_MGTUgd_ASB,
            'CARD':CARD,
            'Proposed_ASBSEA':Proposed_ASBSEA,
            'MICN':MICN,
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        print(self.model)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
