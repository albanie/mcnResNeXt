function demo_resnext(varargin)
%DEMO_RESNEXT - minimal demo with ResNeXt architecutre
%  DEMO_RESNEXT an image classificaiton demo with a ResNeXt 
%  model
%
%   DEMO_RESNEXT(..., 'option', value, ...) accepts the following
%   options:
%
%   `modelPath`:: ''
%    Path to a valid ResNeXt matconvnet model. If none is provided, a model
%    will be downloaded.
%
%   `gpus`:: []
%    Device on which to run network 
%
%   `wrapper`:: 'dagnn'
%    The matconvnet wrapper to be used (both dagnn and autonn are supported) 
%
% Copyright (C) 2017 Samuel Albanie
% All rights reserved.

  opts.gpu = 3 ;
  opts.modelPath = '' ;
  opts.wrapper = 'dagnn' ;
  opts.labelFile = 'misc/imagenet_class_index.json' ;
  opts = vl_argparse(opts, varargin) ;

  % Load or download an example ResNeXt model:
  modelName = 'resnext_50_32x4d-pt-mcn.mat' ;
  paths = {opts.modelPath, ...
           modelName, ...
           fullfile(vl_rootnn, 'data', 'models-import', modelName)} ;
  ok = find(cellfun(@(x) exist(x, 'file'), paths), 1) ;

  if isempty(ok)
    fprintf('Downloading the ResNeXt model ... this may take a while\n') ;
    opts.modelPath = fullfile(vl_rootnn, 'data/models-import', modelName) ;
    modelDir = fileparts(opts.modelPath) ;
    if ~exist(modelDir, 'dir'), mkdir(modelDir) ; end
    baseUrl = 'http://www.robots.ox.ac.uk/~albanie/models' ;
    url = fullfile(baseUrl, sprintf('/pytorch-imports/%s', modelName)) ;
    urlwrite(url, opts.modelPath) ;
  else
    opts.modelPath = paths{ok} ;
  end

  net = load(opts.modelPath) ; net = dagnn.DagNN.loadobj(net);

  % Add softmax to convert confidences into probabilities
  conf = net.layers(end).outputs{1} ;
  net.addLayer('prob', dagnn.SoftMax(), conf, 'prob', {}) ;

  if strcmp(opts.wrapper, 'autonn') % convert to autonn
    out = Layer.fromDagNN(net, @resnext_autonn_custom_fn) ; net = out{1} ;
  end

  % apply normalisation in the manner used for training
  imsz = net.meta.normalization.imageSize ;
  im = single(imread('peppers.png')) ; % load example image
  data = imresize(im, imsz) ;  data = data / 255 ; 
  imMean = net.meta.normalization.averageImage ;
  imStd = net.meta.normalization.imageStd ;
  data = bsxfun(@minus, data, permute(imMean, [3 2 1])) ;
  data = bsxfun(@rdivide, data, permute(imStd, [3 2 1])) ;

  % Evaluate network either on CPU or GPU and move image if required
  if ~isempty(opts.gpu), gpuSetup(net, opts) ; data = gpuArray(data) ; end

  % tell the network to store the outputs
  % of the prediction layer and do a forward pass
  %net.mode = 'test';
  %net.vars(end).precious = true ;
  switch opts.wrapper
    case 'dagnn', inputs = {{'data', data}} ; net.mode = 'test' ;
    case 'autonn', inputs = {{'data', data} 'test'} ;
  end
  
  % gather the predictions from the network and sort by confidence
  net.eval(inputs{:}) ;
  probs = gather(net.vars(end).value) ;
  [probs, sortedIdx] = sort(probs(:), 'descend') ;

  % map the highest confidence into an imagenet label
  label = net.meta.classes.description{sortedIdx(1)} ; 
  conf = probs(1) ;

  % diplay prediction
  figure ; im = im / 255 ; imagesc(im) ;
  title(sprintf('top cls prediction: %s \n confidence: %f', label, conf)) ;
  if exist('zs_dispFig', 'file'), zs_dispFig ; end

  % free up the GPU allocation
  if numel(opts.gpu) > 0, net.move('cpu') ; end

% ---------------------------
function gpuSetup(net, opts)
% --------------------------
  gpuDevice(opts.gpu) ; 
  switch opts.wrapper
    case 'dagnn', onGpu = strcmp(net.device, 'gpu') ;
    case 'autonn', onGpu = net.gpu ;
  end
  if ~onGpu, net.move('gpu') ; end
