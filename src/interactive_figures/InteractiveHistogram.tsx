import { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import { BLUE_RGB, floatYearToDate, Forecast, ORANGE_RGB } from "./utils";

import swebench_forecast from '../assets/data/swebench_forecast.json';
import swebench_forecast_elicited from '../assets/data/swebench_forecast_elicited.json';
import cybench_forecast from '../assets/data/cybench_forecast.json';
import cybench_forecast_elicited from '../assets/data/cybench_forecast_elicited.json';
import rebench_forecast from '../assets/data/rebench_forecast.json';

function InteractiveHistogram() {
    return <InnerHistogram
        plots={[
            {
                title: "SWE-Bench Verified",
                yaxis: "Success Rate",
                defaultCondition: 0.9,
                forecast: swebench_forecast,
                elicited: swebench_forecast_elicited,
            },
            {
                title: "CyBench",
                yaxis: "Success Rate",
                defaultCondition: 0.9,
                forecast: cybench_forecast,
                elicited: cybench_forecast_elicited,
            },
            {
                title: "RE-Bench",
                yaxis: "Score",
                defaultCondition: 1,
                forecast: rebench_forecast,
            }
        ]}
    />
}

function getFirstCrossing(x: number[], y: number[], threshold: number): number | undefined {
    for (let i = 0; i < y.length; i++) {
        if (y[i] >= threshold) {
            return x[i];
        }
    }
    return undefined;
}


type HistogramPlotProps = {
    plots: Array<{
        title: string,
        yaxis: string,
        defaultCondition: number,
        forecast: Forecast,
        elicited?: Forecast,
    }>,
}

function InnerHistogram(props: HistogramPlotProps) {
    const [windowWidth, setWindowWidth] = useState(window.innerWidth);

    useEffect(() => {
        const handleResize = () => setWindowWidth(window.innerWidth);
        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    const plotWidth = Math.min(windowWidth - 20, 1200);
    const plotHeight = 300;

    return (
        <div>
            {props.plots.map((plot, plotIndex) => {

                // get closest idx to default condition in y_linspace
                let defaultIdx = 0;
                for (let i = 0; i < plot.forecast.y_linspace.length; i++) {
                    if (plot.forecast.y_linspace[i] >= plot.defaultCondition) {
                        defaultIdx = i;
                        break;
                    }
                }

                const maxV = Math.max(
                    Math.max(...plot.forecast.density.map(r => r[defaultIdx])),
                    plot.elicited ? Math.max(...plot.elicited.density.map(r => r[defaultIdx])) : 0
                )

                const unelicitedForecast = plot.forecast;
                const densityTraces: Plotly.Data[] = [
                    {
                        type: 'scatter',
                        x: floatYearToDate(unelicitedForecast.x_linspace),
                        y: unelicitedForecast.density.map(r => r[defaultIdx]),
                        name: 'Unelicited',
                        fill: 'tozeroy',
                        line: { color: BLUE_RGB },
                        fillcolor: `${BLUE_RGB}40`
                    }
                ];

                const forecastLines: Plotly.Data[] = [];
                const forecast_pred_x = getFirstCrossing(unelicitedForecast.x_linspace, unelicitedForecast.forecast, plot.defaultCondition);
                if (forecast_pred_x !== undefined) {
                    forecastLines.push({
                        type: 'scatter',
                        x: floatYearToDate([forecast_pred_x, forecast_pred_x]),
                        y: [0, maxV],
                        line: { color: BLUE_RGB, },
                    });
                }


                if (plot.elicited) {
                    const elicitedForecast = plot.elicited;
                    densityTraces.push({
                        type: 'scatter',
                        x: floatYearToDate(plot.elicited.x_linspace),
                        y: plot.elicited.density.map(r => r[defaultIdx]),
                        name: 'Elicited',
                        fill: 'tozeroy',
                        line: { color: ORANGE_RGB },
                        fillcolor: `${ORANGE_RGB}40`
                    });
                    const forecast_pred_x = getFirstCrossing(elicitedForecast.x_linspace, elicitedForecast.forecast, plot.defaultCondition);
                    if (forecast_pred_x !== undefined) {
                        forecastLines.push({
                            type: 'scatter',
                            x: floatYearToDate([forecast_pred_x, forecast_pred_x]),
                            y: [0, maxV],
                            line: { color: ORANGE_RGB, }
                        });
                    }
                }

                const slider: Partial<Plotly.Slider> = {
                    active: defaultIdx,
                    currentvalue: {
                        visible: true,
                        prefix: plot.yaxis + ': ',
                        xanchor: 'center',
                    },
                    steps: unelicitedForecast.y_linspace.map((val, j) => {
                        const maxV = Math.max(
                            Math.max(...plot.forecast.density.map(r => r[j])),
                            plot.elicited ? Math.max(...plot.elicited.density.map(r => r[j])) : 0
                        )

                        const forecast_pred_x_unelicited = getFirstCrossing(unelicitedForecast.x_linspace, unelicitedForecast.forecast, val) ?? 0;

                        const forecast_pred_x_elicited = plot.elicited ? getFirstCrossing(plot.elicited.x_linspace, plot.elicited.forecast, val) ?? 0 : 0;

                        return {
                            label: val.toFixed(2),
                            method: 'update',
                            args: [
                                {
                                    'x': plot.elicited
                                        ? [
                                            floatYearToDate(unelicitedForecast.x_linspace),
                                            floatYearToDate(plot.elicited.x_linspace),
                                            floatYearToDate([forecast_pred_x_unelicited, forecast_pred_x_unelicited]),
                                            floatYearToDate([forecast_pred_x_elicited, forecast_pred_x_elicited])
                                        ]
                                        : [
                                            floatYearToDate(unelicitedForecast.x_linspace),
                                            floatYearToDate([forecast_pred_x_unelicited, forecast_pred_x_unelicited])
                                        ],
                                    'y': plot.elicited
                                        ? [
                                            unelicitedForecast.density.map(r => r[j]),
                                            plot.elicited.density.map(r => r[j]),
                                            [0, maxV],
                                            [0, maxV]
                                        ]
                                        : [
                                            unelicitedForecast.density.map(r => r[j]),
                                            [0, maxV],
                                        ],
                                },
                                {
                                    'xaxis.type': 'date',
                                }
                            ]
                        }
                    }),
                    len: 0.5,
                    x: 0.25,
                    y: 0,
                    pad: { t: 30 },
                    ticklen: 4,
                    minorticklen: 0,
                };

                return (
                    <Plot
                        key={plotIndex}
                        data={[...densityTraces, ...forecastLines]}
                        layout={{
                            width: plotWidth,
                            height: plotHeight,
                            showlegend: false,
                            margin: { t: 40, b: 20, l: 0, r: 0 },
                            sliders: [slider],
                            xaxis: { type: 'date' },
                            title: {
                                text: plot.title,
                                font: { size: 24 }
                            }
                        }}
                        config={{ scrollZoom: false, displayModeBar: false, responsive: true }}
                    />
                );
            })}
        </div>
    );
}

export default InteractiveHistogram;