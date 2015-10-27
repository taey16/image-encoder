
-- global
-- datasetTrain
-- datasetVal
paths.dofile(opt.data_loader_file)
datasetTrain, datasetVal = createDatasetsDistorted(opt.dataset_root, opt.batchsize, opt.test_batchsize)
print('===> Loaded dataset: ' .. opt.data_loader_file) 

