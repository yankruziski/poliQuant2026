// chart-config.js — Layout e configuracao visual do grafico 3D Monte Carlo

const COLORSCALE = [
  [0.0,  "rgb(8,29,88)"],
  [0.15, "rgb(37,52,148)"],
  [0.3,  "rgb(34,94,168)"],
  [0.45, "rgb(29,145,192)"],
  [0.6,  "rgb(65,182,196)"],
  [0.75, "rgb(161,218,180)"],
  [0.9,  "rgb(255,255,204)"],
  [1.0,  "rgb(255,237,160)"],
];

function buildLayout(titulo, subtitulo) {
  return {
    title: {
      text: titulo + "<br><sub>" + subtitulo + "</sub>",
      font: { size: 18, color: "white" },
    },
    paper_bgcolor: "rgb(17,17,17)",
    plot_bgcolor: "rgb(17,17,17)",
    scene: {
      xaxis_title: "Dia",
      yaxis_title: "Retorno (%)",
      zaxis_title: "Densidade",
      xaxis: {
        backgroundcolor: "rgb(25,25,25)",
        gridcolor: "rgb(60,60,60)",
        color: "white",
      },
      yaxis: {
        backgroundcolor: "rgb(25,25,25)",
        gridcolor: "rgb(60,60,60)",
        color: "white",
      },
      zaxis: {
        backgroundcolor: "rgb(25,25,25)",
        gridcolor: "rgb(60,60,60)",
        color: "white",
      },
      camera: { eye: { x: 1.8, y: -1.5, z: 0.8 } },
    },
    legend: { font: { color: "white" } },
    autosize: true,
    margin: { l: 0, r: 0, t: 80, b: 0 },
  };
}
