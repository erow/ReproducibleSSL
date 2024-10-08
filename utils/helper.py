import argparse, os
from pathlib import Path
import gin

@gin.configurable()
def build_model(args,model_fn):    
    model = model_fn()
    return model


def aug_parse(parser: argparse.ArgumentParser):
    import yaml
    aug_parser = argparse.ArgumentParser(add_help=False)
    aug_parser.add_argument('--cfgs', nargs='+', default=[],
                        help='<Required> Config files *.gin.', required=False)
    aug_parser.add_argument('--gin', nargs='+', default=[],
                        help='Overrides config values. e.g. --gin "section.option=value"')
    aug_parser.add_argument('--config', type=str, default=None, help='The config file path')
        
    aug_args, unkowns = aug_parser.parse_known_args()

    # override parser defaults
    if aug_args.config:
        with open(aug_args.config) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        for k,v in config.items():
            assert k != "config", "config is a reserved key"
            assert k != "cfgs", "cfgs is a reserved key"
            if k == "gin":
                if v:
                    aug_args.gin = v + aug_args.gin
            else:
                parser.set_defaults(**{k:v})
            

    gin.parse_config_files_and_bindings(aug_args.cfgs, aug_args.gin)

    # save config
    args, unkowns = parser.parse_known_args()
    print("warn! unknown args: ", unkowns)
    if args.output_dir:
        output_dir=Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(args.output_dir,'config.yml'), 'w') as f:
            yaml.dump(vars(args), f)
            
        open(output_dir/"config.gin",'w').write(gin.config_str(),)
    
    return args