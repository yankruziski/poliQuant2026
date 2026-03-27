// chart-render.js — Monta os traces e renderiza o grafico 3D

document.addEventListener("DOMContentLoaded", function () {
  if (typeof CHART_DATA === "undefined") {
    console.error("chart-data.js nao carregado");
    return;
  }

  var d = CHART_DATA;

  var surfaceTrace = {
    type: "surface",
    x: d.x_dias,
    y: d.y_grid,
    z: d.z_surface,
    colorscale: COLORSCALE,
    opacity: 0.92,
    contours: {
      z: {
        show: true,
        usecolormap: true,
        highlightcolor: "white",
        project_z: true,
      },
    },
    hovertemplate:
      "Dia: %{x}<br>Retorno: %{y:.1f}%<br>Densidade: %{z:.4f}<extra></extra>",
  };

  var medianaTrace = {
    type: "scatter3d",
    x: d.mediana_dias,
    y: d.mediana_ret,
    z: d.mediana_z,
    mode: "lines",
    line: { color: "#00ff88", width: 6 },
    name: "Mediana",
  };

  var layout = buildLayout(d.titulo, d.subtitulo);

  Plotly.newPlot("chart", [surfaceTrace, medianaTrace], layout, {
    responsive: true,
  });
});
