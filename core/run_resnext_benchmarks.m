function run_resnext_benchmarks
% do a single pass over the imagenet validation data

gpus = 3 ;
batchSize = 256 ;
useCached = 1 ; % load results from cache if available

importedModels = {
  'resnext_50_32x4d-pt-mcn', ...
  'resnext_101_32x4d-pt-mcn', ...
  'resnext_101_64x4d-pt-mcn', ...
} ;

for ii = 1:numel(importedModels)
  model = importedModels{ii} ;
  imagenet_eval(model, batchSize, gpus, useCached) ;
end

% -------------------------------------------------------
function imagenet_eval(model, batchSize, gpus, useCached)
% -------------------------------------------------------
[~,info] = resnext_imagenet('model', model, 'batchSize', ...
               batchSize, 'gpus', gpus, 'continue', useCached) ;
top1 = info.val.top1err * 100 ; top5 = info.val.top5err * 100 ;
fprintf('%s: top-1: %.2f, top-5: %.2f\n', model, top1, top5) ;
