function params = setParams()
% set parameters

params.glider.M     = 3.366;    % aircraft mass                 (kg)
params.glider.S     = .568;     % reference area                (m2)
params.glider.ST    = 10.2;     % aspect ratio                  (-)
params.glider.OE    = .9;       % oswald factor                 (-)
params.glider.CD0   = .015;     % zero lift drag coefficient    (-)

params.physics.RHO  = 1.225;    % air density                   (kg/m3)
params.physics.G    = 9.81;     % gravitational acceleration    (m/s2)

end
