import os
import torch
import argparse

from transformers import AutoConfig, AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dpr_model', required=True)
    parser.add_argument('--save_to', required=True)
    args = parser.parse_args()

    dpr_model_ckpt = torch.load(args.dpr_model, map_location='cpu')
    config_name = dpr_model_ckpt['encoder_params']['pretrained_model_cfg']
    dpr_model_dict = dpr_model_ckpt['model_dict']

    AutoConfig.from_pretrained(config_name).save_pretrained(args.save_to)
    AutoTokenizer.from_pretrained(config_name).save_pretrained(args.save_to)

    question_keys = [k for k in dpr_model_dict.keys() if k.startswith('question_model')]
    ctx_keys = [k for k in dpr_model_dict.keys() if k.startswith('ctx_model')]

    question_dict = dict([(k[len('question_model')+1:], dpr_model_dict[k]) for k in question_keys])
    ctx_dict = dict([(k[len('ctx_model')+1:], dpr_model_dict[k]) for k in ctx_keys])

    os.makedirs(os.path.join(args.save_to, 'query_model'), exist_ok=True)
    os.makedirs(os.path.join(args.save_to, 'passage_model'), exist_ok=True)
    torch.save(question_dict, os.path.join(args.save_to, 'query_model', 'pytorch_model.bin'))
    torch.save(ctx_dict, os.path.join(args.save_to, 'passage_model', 'pytorch_model.bin'))

    
if __name__ == '__main__':
    main()