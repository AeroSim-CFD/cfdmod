# import sys

# if "--virtual-env" in sys.argv:
#     virtualEnvPath = sys.argv[sys.argv.index("--virtual-env") + 1]
#     virtualEnv = virtualEnvPath + "/bin/activate_this.py"
#     if sys.version_info.major < 3:
#         execfile(virtualEnv, dict(__file__=virtualEnv))
#     else:
#         exec(open(virtualEnv).read(), {"__file__": virtualEnv})

# Your script continues here, now with access to packages in the virtual environment
import pandas as pd
from paraview.simple import *

reader = XMLPolyDataReader(FileName="input.vtp")

# view = GetActiveView() if not view else CreateRenderView()
view = CreateRenderView()


view.CameraViewUp = [0, 0, 1]
view.CameraFocalPoint = [0, 0, 0]
view.CameraViewAngle = 45
view.CameraPosition = [5, 0, 0]

Show()

view.Background = [1, 1, 1]  # white

view.ViewSize = [200, 300]  # [width, height]

dp = GetDisplayProperties()

dp.AmbientColor = [1, 0, 0]  # red

dp.DiffuseColor = [0, 1, 0]  # blue

dp.PointSize = 2

dp.Representation = "Surface"

Render()

WriteImage("test.png")
