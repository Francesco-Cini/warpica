function result = isgpuarray(value)
% CPU-only compatibility for MATLAB installations without Parallel Computing Toolbox.
% The numerical upstream code is unchanged. Never use this shim for GPU runs.
result = isa(value, 'gpuArray');
end
