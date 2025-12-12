python - << 'EOF'
import pyvista as pv
mesh = pv.read("./brain_data/closed_solid/output.ply")
mesh.plot()
EOF

python - << 'EOF'
import pyvista as pv
mesh = pv.read("./brain_data/closed_solid/output_scalar_fields.ply")
mesh.plot(rgb=True)
EOF
