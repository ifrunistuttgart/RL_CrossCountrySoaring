function [alpha_bestGlide, V_bestGlide, glideRatio] = getBestGlide()
% compute best glide

params = setParams();

alpha_bestGlide = ((params.glider.ST + 2)...
    * sqrt(params.glider.CD0*params.glider.OE/params.glider.ST))...
    /(2*sqrt(pi));

cL_bestGlide = (2*pi*alpha_bestGlide*params.glider.ST)...
    /(params.glider.ST + 2);

V_bestGlide = sqrt((2*params.glider.M*params.physics.G)...
    /(params.physics.RHO*params.glider.S*cL_bestGlide));

cD_bestGlide = params.glider.CD0 + (1/(pi*params.glider.ST*params.glider.OE))*cL_bestGlide^2;

glideRatio = cL_bestGlide/cD_bestGlide;
end