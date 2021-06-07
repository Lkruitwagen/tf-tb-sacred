import click, os, logging
from click import command, option, Option, UsageError
logging.basicConfig(level=logging.INFO)

@click.group()
def cli():
    pass

@cli.command()
@click.option('--conf', default=os.path.join(os.getcwd(),'conf.yaml'), help='path to conf.yaml')
def train(conf_path):
    """
    Train model on VOC2012 data.
    
    \b
    PARAMETERS
    ----------
    <none>
    """
    from myproject.main import ex
    
    ex.add_config()
    
    ex.run()
        
if __name__=="__main__":
    cli()