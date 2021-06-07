import os, subprocess, logging
logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger('downloader')

root = os.getcwd()

logger.info('Downloading...')
### Let's download our data using wget same as we did for Flowers. This is a bigger dataset (~2gb) so be warned!
p = subprocess.Popen(['wget','-c','https://storage.googleapis.com/voc-mirror/voc2012.tar', '-O',f'{root}/data/voc2012.tar'])
p.wait()

### Now let's untar our data. (note: specify the directory with -C, and untar options with -x (extract),-v (verbose),-f (pass filename),-z (gzip))
logger.info('Untarring')
p = subprocess.Popen(['tar','-C',f'{root}/data','-xf',f'{root}/data/voc2012.tar'])
p.wait()

### Cleanup - remove our original .tar file.
logger.info('Cleanup - removing .tar')
os.remove(f'{root}/data/voc2012.tar')

logger.info('Done!')