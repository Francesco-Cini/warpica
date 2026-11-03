function export_matlab_reference(upstreamRoot, outputFile)
% Reproduce fixtures using unmodified WarpFactory at the pinned commit.
addpath(genpath(upstreamRoot));
% Always use the CPU helpers so the export does not depend on whether the
% machine has the Parallel Computing Toolbox or a supported GPU.
addpath(fullfile(fileparts(mfilename('fullpath')),'matlab_cpu_compat'),'-begin');
output.commit = '03b10cb02e73998af28db87201a43f2fa0e30319';
output.runtime = version;
output.generated_utc = char(datetime('now','TimeZone','UTC'));
output.c = c;
output.G = G;
spacing = [0.25/c, 0.5, 0.5, 0.5];

% Each case: name, grid, offset added to the default centre, generator.
% Comoving generators need a single time slice. The three original cases
% keep their exact parameters.
cases = struct('name',{},'grid',{},'offset',{},'make',{});
cases = addCase(cases,'minkowski',[5,5,5,5],[0,0,0,0], ...
    @(g,ctr,s) metricGet_Minkowski(g,s));
cases = addCase(cases,'alcubierre_static',[1,5,5,5],[0,0,0,0], ...
    @(g,ctr,s) metricGet_Alcubierre(g,ctr,0.2,0.8,2,s));
cases = addCase(cases,'alcubierre_dynamic',[5,5,5,5],[0,0,0,0], ...
    @(g,ctr,s) metricGet_Alcubierre(g,ctr,0.2,0.8,2,s));
cases = addCase(cases,'alcubierre_comoving',[1,5,5,5],[0,0,0,0], ...
    @(g,ctr,s) metricGet_AlcubierreComoving(g,ctr,0.2,0.8,2,s));
cases = addCase(cases,'lentz_dynamic',[5,5,5,5],[0,0,0,0], ...
    @(g,ctr,s) metricGet_Lentz(g,ctr,0.2,1,s));
cases = addCase(cases,'lentz_comoving',[1,5,5,5],[0,0,0,0], ...
    @(g,ctr,s) metricGet_LentzComoving(g,ctr,0.2,1,s));
% Modified Time is deliberately absent: upstream metricGet_ModifiedTime and
% metricGet_ModifiedTimeComoving are defective at the pinned commit (see
% docs/reference-validation.md), so neither can serve as a reference.
cases = addCase(cases,'van_den_broeck_dynamic',[5,5,5,5],[0,0,0,0], ...
    @(g,ctr,s) metricGet_VanDenBroeck(g,ctr,0.2,1,1,2,1,0.1,s));
% Offset the centre so no grid point sits at the singular origin.
cases = addCase(cases,'schwarzschild',[1,5,5,5],[0,0.13,0.11,0.07], ...
    @(g,ctr,s) metricGet_Schwarzschild(g,ctr,0.05,s));
cases = addCase(cases,'warp_shell_comoving',[1,5,5,5],[0,0,0,0], ...
    @(g,ctr,s) metricGet_WarpShellComoving(g,ctr,2e26,0.5,1.5,0.1,1,1,0.2,true,s));

for ci = 1:numel(cases)
    name = cases(ci).name;
    grid = cases(ci).grid;
    center = (grid+1)/2.*spacing + cases(ci).offset;
    metric = cases(ci).make(grid,center,spacing);
    item = struct();
    item.grid = grid; item.spacing = spacing; item.center = center;
    item.metric = packTensor(metric.tensor,grid);
    inverse = c4Inv(metric.tensor);
    item.ricci_second = packTensor(ricciT2(inverse,metric.tensor,spacing),grid);
    item.ricci_fourth = packTensor(ricciT(inverse,metric.tensor,spacing),grid);
    energy2 = getEnergyTensor(metric,0,'second');
    energy4 = getEnergyTensor(metric,0,'fourth');
    item.energy_second = packTensor(energy2.tensor,grid);
    item.energy_fourth = packTensor(energy4.tensor,grid);
    frame = doFrameTransfer(metric,energy4,'Eulerian');
    item.eulerian = packTensor(frame.tensor,grid);
    [expansion,shear,vorticity] = getScalars(metric);
    item.expansion = packArray(expansion,grid);
    item.shear = packArray(shear,grid);
    item.vorticity = packArray(vorticity,grid);
    item.null = packArray(getEnergyConditions(energy4,metric,'Null',12,3,0),grid);
    item.weak = packArray(getEnergyConditions(energy4,metric,'Weak',12,3,0),grid);
    item.strong = packArray(getEnergyConditions(energy4,metric,'Strong',12,3,0),grid);
    item.dominant = packArray(getEnergyConditions(energy4,metric,'Dominant',12,3,0),grid);
    output.(name) = item;
    fprintf('Exported case %s\n',name);
end
fid=fopen(outputFile,'w');
assert(fid ~= -1,'Cannot open output file');
cleanup=onCleanup(@() fclose(fid));
fwrite(fid,jsonencode(output),'char');
fprintf('Exported independent MATLAB reference to %s\n',outputFile);
end

function cases = addCase(cases,name,grid,offset,make)
cases(end+1).name = name;
cases(end).grid = grid;
cases(end).offset = offset;
cases(end).make = make;
end

function packed = packTensor(tensor,grid)
array=zeros([4,4,grid]);
for i=1:4
    for j=1:4
        array(i,j,:,:,:,:) = reshape(tensor{i,j},[1,1,grid]);
    end
end
packed=packArray(array,[4,4,grid]);
end

function packed = packArray(array,shape)
packed.shape=shape;
packed.data=array(:);
end
