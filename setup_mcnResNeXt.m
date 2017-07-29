function setup_mcnResNeXt()
%SETUP_MCNRFCN Sets up mcnResNeXt, by adding its folders 
% to the Matlab path

  root = fileparts(mfilename('fullpath')) ;
  addpath(root, [root '/matlab'], [root '/core']) ;
