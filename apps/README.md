This script assumes that the coder data is saved in the  ../data directory. The default location of the coder_data's data directory can be changed as below,

`
hcmi = cd.DatasetLoader('hcmi', DATA_DIRECTORY)
beataml = cd.DatasetLoader('beataml', DATA_DIRECTORY)
cptac = cd.DatasetLoader('cptac', DATA_DIRECTORY)
depmap = cd.DatasetLoader('broad_sanger', DATA_DIRECTORY)
mpnst = cd.DatasetLoader('mpnst', DATA_DIRECTORY)
`
