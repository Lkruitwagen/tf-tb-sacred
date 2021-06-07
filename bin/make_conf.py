import yaml, os

params = dict(
    data_root=os.path.join(os.getcwd(),'data'), 
    input_shape=(224,224), 
    finetune_encoder=False, 
    trn_split=0.7, 
    val_split=0.85, 
    generator_mode='resize', 
    batch_size=32, 
    n_epochs=20
)

yaml.dump(params, open(os.path.join(os.getcwd(),'config.yaml'),'w'))