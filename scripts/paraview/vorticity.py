# trace generated using paraview version 5.10.0-RC1
# import paraview
# paraview.compatibility.major = 5
# paraview.compatibility.minor = 10

#### import the simple module from the paraview
from paraview.simple import *

#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# find source
macrs_0050000vtm = FindSource("macrs_0000000.vtm*")

# create a new 'Calculator'
calculator1 = Calculator(registrationName="Calculator1", Input=macrs_0050000vtm)
calculator1.AttributeType = "Cell Data"
calculator1.Function = ""

# Properties modified on calculator1
calculator1.ResultArrayName = "Velocity"
calculator1.Function = "ux*iHat + uy*jHat + uz*kHat"

# get active view
renderView1 = GetActiveViewOrCreate("RenderView")

# show data in view
calculator1Display = Show(calculator1, renderView1, "UniformGridRepresentation")

# trace defaults for the display properties.
calculator1Display.Representation = "Outline"
calculator1Display.ColorArrayName = [None, ""]
calculator1Display.SelectTCoordArray = "None"
calculator1Display.SelectNormalArray = "None"
calculator1Display.SelectTangentArray = "None"
calculator1Display.OSPRayScaleFunction = "PiecewiseFunction"
calculator1Display.SelectOrientationVectors = "Velocity"
calculator1Display.ScaleFactor = 20.0
calculator1Display.SelectScaleArray = "None"
calculator1Display.GlyphType = "Arrow"
calculator1Display.GlyphTableIndexArray = "None"
calculator1Display.GaussianRadius = 1.0
calculator1Display.SetScaleArray = [None, ""]
calculator1Display.ScaleTransferFunction = "PiecewiseFunction"
calculator1Display.OpacityArray = [None, ""]
calculator1Display.OpacityTransferFunction = "PiecewiseFunction"
calculator1Display.DataAxesGrid = "GridAxesRepresentation"
calculator1Display.PolarAxes = "PolarAxesRepresentation"
calculator1Display.ScalarOpacityUnitDistance = 1.5662881898411756
calculator1Display.OpacityArrayName = ["CELLS", "Velocity"]
calculator1Display.SliceFunction = "Plane"
calculator1Display.Slice = 24

# init the 'Plane' selected for 'SliceFunction'
calculator1Display.SliceFunction.Origin = [100.0, 48.0, 24.0]

# hide data in view
Hide(macrs_0050000vtm, renderView1)

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
ColorBy(calculator1Display, ("FIELD", "vtkBlockColors"))

# show color bar/color legend
calculator1Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'vtkBlockColors'
vtkBlockColorsLUT = GetColorTransferFunction("vtkBlockColors")

# get opacity transfer function/opacity map for 'vtkBlockColors'
vtkBlockColorsPWF = GetOpacityTransferFunction("vtkBlockColors")

# create a new 'Cell Data to Point Data'
cellDatatoPointData1 = CellDatatoPointData(
    registrationName="CellDatatoPointData1", Input=calculator1
)
cellDatatoPointData1.CellDataArraytoprocess = ["Velocity", "rho", "ux", "uy", "uz"]

# Properties modified on cellDatatoPointData1
cellDatatoPointData1.ProcessAllArrays = 0
cellDatatoPointData1.CellDataArraytoprocess = ["Velocity"]

# show data in view
cellDatatoPointData1Display = Show(cellDatatoPointData1, renderView1, "UniformGridRepresentation")

# trace defaults for the display properties.
cellDatatoPointData1Display.Representation = "Outline"
cellDatatoPointData1Display.ColorArrayName = [None, ""]
cellDatatoPointData1Display.SelectTCoordArray = "None"
cellDatatoPointData1Display.SelectNormalArray = "None"
cellDatatoPointData1Display.SelectTangentArray = "None"
cellDatatoPointData1Display.OSPRayScaleArray = "Velocity"
cellDatatoPointData1Display.OSPRayScaleFunction = "PiecewiseFunction"
cellDatatoPointData1Display.SelectOrientationVectors = "None"
cellDatatoPointData1Display.ScaleFactor = 20.0
cellDatatoPointData1Display.SelectScaleArray = "None"
cellDatatoPointData1Display.GlyphType = "Arrow"
cellDatatoPointData1Display.GlyphTableIndexArray = "None"
cellDatatoPointData1Display.GaussianRadius = 1.0
cellDatatoPointData1Display.SetScaleArray = ["POINTS", "Velocity"]
cellDatatoPointData1Display.ScaleTransferFunction = "PiecewiseFunction"
cellDatatoPointData1Display.OpacityArray = ["POINTS", "Velocity"]
cellDatatoPointData1Display.OpacityTransferFunction = "PiecewiseFunction"
cellDatatoPointData1Display.DataAxesGrid = "GridAxesRepresentation"
cellDatatoPointData1Display.PolarAxes = "PolarAxesRepresentation"
cellDatatoPointData1Display.ScalarOpacityUnitDistance = 1.5662881898411756
cellDatatoPointData1Display.OpacityArrayName = ["POINTS", "Velocity"]
cellDatatoPointData1Display.SliceFunction = "Plane"
cellDatatoPointData1Display.Slice = 24

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
cellDatatoPointData1Display.ScaleTransferFunction.Points = [
    -0.016180973732843995,
    0.0,
    0.5,
    0.0,
    0.08600391168147326,
    1.0,
    0.5,
    0.0,
]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
cellDatatoPointData1Display.OpacityTransferFunction.Points = [
    -0.016180973732843995,
    0.0,
    0.5,
    0.0,
    0.08600391168147326,
    1.0,
    0.5,
    0.0,
]

# init the 'Plane' selected for 'SliceFunction'
cellDatatoPointData1Display.SliceFunction.Origin = [100.0, 48.0, 24.0]

# hide data in view
Hide(calculator1, renderView1)

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
ColorBy(cellDatatoPointData1Display, ("FIELD", "vtkBlockColors"))

# show color bar/color legend
cellDatatoPointData1Display.SetScalarBarVisibility(renderView1, True)

# create a new 'Compute Derivatives'
computeDerivatives1 = ComputeDerivatives(
    registrationName="ComputeDerivatives1", Input=cellDatatoPointData1
)
computeDerivatives1.Scalars = [None, ""]
computeDerivatives1.Vectors = ["POINTS", "Velocity"]

# Properties modified on computeDerivatives1
computeDerivatives1.Scalars = ["POINTS", ""]
computeDerivatives1.OutputVectorType = "Vorticity"
computeDerivatives1.OutputTensorType = "Nothing"

# show data in view
computeDerivatives1Display = Show(computeDerivatives1, renderView1, "UniformGridRepresentation")

# trace defaults for the display properties.
computeDerivatives1Display.Representation = "Outline"
computeDerivatives1Display.ColorArrayName = [None, ""]
computeDerivatives1Display.SelectTCoordArray = "None"
computeDerivatives1Display.SelectNormalArray = "None"
computeDerivatives1Display.SelectTangentArray = "None"
computeDerivatives1Display.OSPRayScaleArray = "Velocity"
computeDerivatives1Display.OSPRayScaleFunction = "PiecewiseFunction"
computeDerivatives1Display.SelectOrientationVectors = "Vorticity"
computeDerivatives1Display.ScaleFactor = 20.0
computeDerivatives1Display.SelectScaleArray = "None"
computeDerivatives1Display.GlyphType = "Arrow"
computeDerivatives1Display.GlyphTableIndexArray = "None"
computeDerivatives1Display.GaussianRadius = 1.0
computeDerivatives1Display.SetScaleArray = ["POINTS", "Velocity"]
computeDerivatives1Display.ScaleTransferFunction = "PiecewiseFunction"
computeDerivatives1Display.OpacityArray = ["POINTS", "Velocity"]
computeDerivatives1Display.OpacityTransferFunction = "PiecewiseFunction"
computeDerivatives1Display.DataAxesGrid = "GridAxesRepresentation"
computeDerivatives1Display.PolarAxes = "PolarAxesRepresentation"
computeDerivatives1Display.ScalarOpacityUnitDistance = 1.5662881898411756
computeDerivatives1Display.OpacityArrayName = ["POINTS", "Velocity"]
computeDerivatives1Display.SliceFunction = "Plane"
computeDerivatives1Display.Slice = 24

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
computeDerivatives1Display.ScaleTransferFunction.Points = [
    -0.016180973732843995,
    0.0,
    0.5,
    0.0,
    0.08600391168147326,
    1.0,
    0.5,
    0.0,
]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
computeDerivatives1Display.OpacityTransferFunction.Points = [
    -0.016180973732843995,
    0.0,
    0.5,
    0.0,
    0.08600391168147326,
    1.0,
    0.5,
    0.0,
]

# init the 'Plane' selected for 'SliceFunction'
computeDerivatives1Display.SliceFunction.Origin = [100.0, 48.0, 24.0]

# hide data in view
Hide(cellDatatoPointData1, renderView1)

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
ColorBy(computeDerivatives1Display, ("FIELD", "vtkBlockColors"))

# show color bar/color legend
computeDerivatives1Display.SetScalarBarVisibility(renderView1, True)

# create a new 'Cell Data to Point Data'
cellDatatoPointData2 = CellDatatoPointData(
    registrationName="CellDatatoPointData2", Input=computeDerivatives1
)
cellDatatoPointData2.CellDataArraytoprocess = ["Vorticity"]

# Properties modified on cellDatatoPointData2
cellDatatoPointData2.ProcessAllArrays = 0

# show data in view
cellDatatoPointData2Display = Show(cellDatatoPointData2, renderView1, "UniformGridRepresentation")

# trace defaults for the display properties.
cellDatatoPointData2Display.Representation = "Outline"
cellDatatoPointData2Display.ColorArrayName = [None, ""]
cellDatatoPointData2Display.SelectTCoordArray = "None"
cellDatatoPointData2Display.SelectNormalArray = "None"
cellDatatoPointData2Display.SelectTangentArray = "None"
cellDatatoPointData2Display.OSPRayScaleArray = "Velocity"
cellDatatoPointData2Display.OSPRayScaleFunction = "PiecewiseFunction"
cellDatatoPointData2Display.SelectOrientationVectors = "None"
cellDatatoPointData2Display.ScaleFactor = 20.0
cellDatatoPointData2Display.SelectScaleArray = "None"
cellDatatoPointData2Display.GlyphType = "Arrow"
cellDatatoPointData2Display.GlyphTableIndexArray = "None"
cellDatatoPointData2Display.GaussianRadius = 1.0
cellDatatoPointData2Display.SetScaleArray = ["POINTS", "Velocity"]
cellDatatoPointData2Display.ScaleTransferFunction = "PiecewiseFunction"
cellDatatoPointData2Display.OpacityArray = ["POINTS", "Velocity"]
cellDatatoPointData2Display.OpacityTransferFunction = "PiecewiseFunction"
cellDatatoPointData2Display.DataAxesGrid = "GridAxesRepresentation"
cellDatatoPointData2Display.PolarAxes = "PolarAxesRepresentation"
cellDatatoPointData2Display.ScalarOpacityUnitDistance = 1.5662881898411756
cellDatatoPointData2Display.OpacityArrayName = ["POINTS", "Velocity"]
cellDatatoPointData2Display.SliceFunction = "Plane"
cellDatatoPointData2Display.Slice = 24

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
cellDatatoPointData2Display.ScaleTransferFunction.Points = [
    -0.016180973732843995,
    0.0,
    0.5,
    0.0,
    0.08600391168147326,
    1.0,
    0.5,
    0.0,
]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
cellDatatoPointData2Display.OpacityTransferFunction.Points = [
    -0.016180973732843995,
    0.0,
    0.5,
    0.0,
    0.08600391168147326,
    1.0,
    0.5,
    0.0,
]

# init the 'Plane' selected for 'SliceFunction'
cellDatatoPointData2Display.SliceFunction.Origin = [100.0, 48.0, 24.0]

# hide data in view
Hide(computeDerivatives1, renderView1)

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
ColorBy(cellDatatoPointData2Display, ("FIELD", "vtkBlockColors"))

# show color bar/color legend
cellDatatoPointData2Display.SetScalarBarVisibility(renderView1, True)

# create a new 'Calculator'
calculator2 = Calculator(registrationName="Calculator2", Input=cellDatatoPointData2)
calculator2.Function = ""

# Properties modified on calculator2
calculator2.ResultArrayName = "IsoSurf"
calculator2.Function = "mag(Vorticity)"

# show data in view
calculator2Display = Show(calculator2, renderView1, "UniformGridRepresentation")

# trace defaults for the display properties.
calculator2Display.Representation = "Outline"
calculator2Display.ColorArrayName = ["POINTS", ""]
calculator2Display.SelectTCoordArray = "None"
calculator2Display.SelectNormalArray = "None"
calculator2Display.SelectTangentArray = "None"
calculator2Display.OSPRayScaleArray = "IsoSurf"
calculator2Display.OSPRayScaleFunction = "PiecewiseFunction"
calculator2Display.SelectOrientationVectors = "None"
calculator2Display.ScaleFactor = 20.0
calculator2Display.SelectScaleArray = "IsoSurf"
calculator2Display.GlyphType = "Arrow"
calculator2Display.GlyphTableIndexArray = "IsoSurf"
calculator2Display.GaussianRadius = 1.0
calculator2Display.SetScaleArray = ["POINTS", "IsoSurf"]
calculator2Display.ScaleTransferFunction = "PiecewiseFunction"
calculator2Display.OpacityArray = ["POINTS", "IsoSurf"]
calculator2Display.OpacityTransferFunction = "PiecewiseFunction"
calculator2Display.DataAxesGrid = "GridAxesRepresentation"
calculator2Display.PolarAxes = "PolarAxesRepresentation"
calculator2Display.ScalarOpacityUnitDistance = 1.5662881898411756
calculator2Display.OpacityArrayName = ["POINTS", "IsoSurf"]
calculator2Display.IsosurfaceValues = [0.04951843770150585]
calculator2Display.SliceFunction = "Plane"
calculator2Display.Slice = 24

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
calculator2Display.ScaleTransferFunction.Points = [
    0.0,
    0.0,
    0.5,
    0.0,
    0.0990368754030117,
    1.0,
    0.5,
    0.0,
]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
calculator2Display.OpacityTransferFunction.Points = [
    0.0,
    0.0,
    0.5,
    0.0,
    0.0990368754030117,
    1.0,
    0.5,
    0.0,
]

# init the 'Plane' selected for 'SliceFunction'
calculator2Display.SliceFunction.Origin = [100.0, 48.0, 24.0]

# hide data in view
Hide(cellDatatoPointData2, renderView1)

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
ColorBy(calculator2Display, ("FIELD", "vtkBlockColors"))

# show color bar/color legend
calculator2Display.SetScalarBarVisibility(renderView1, True)

# ================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
# ================================================================

# get layout
layout1 = GetLayout()

# --------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(1542, 784)

# -----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.CameraPosition = [100.0, 48.0, 462.49202684421596]
renderView1.CameraFocalPoint = [100.0, 48.0, 24.0]
renderView1.CameraParallelScale = 113.49008767288886

# --------------------------------------------
# uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
