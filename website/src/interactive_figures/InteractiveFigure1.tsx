import Plot from 'react-plotly.js'
import agentic_benchmark from '../assets/data/agentic_benchmark.json';
import cybench_scaffolds from '../assets/data/cybench_scaffolds.json';
import swebench_scaffolds from '../assets/data/swebench_scaffolds.json';
import swebench_observed_future_data from '../assets/data/swebench_observed_future_data.json';
import swebench_forecast from '../assets/data/swebench_forecast.json';
import swebench_forecast_elicited from '../assets/data/swebench_forecast_elicited.json';
import cybench_forecast from '../assets/data/cybench_forecast.json';
import cybench_forecast_elicited from '../assets/data/cybench_forecast_elicited.json';
import rebench_forecast from '../assets/data/rebench_forecast.json';
import { useState, useEffect } from 'react';
import { BLUE_RGB, RED_RGB, floatYearToDate, Forecast, ORANGE_RGB, get_frontier_model_set } from './utils';


function InteractiveFigure1(_: {}) {
    return <Fig1 plots={[
        {
            title: "SWE-Bench Verified",
            yaxis: "Success Rate",
            unelicited: {
                scatter: {
                    xpoints: agentic_benchmark.map(r => r["release_date"]),
                    ypoints: agentic_benchmark.map(r => r["SWE-Bench Verified"]),
                    text: agentic_benchmark.map(r => r["model"]),
                },
                forecast: swebench_forecast,
            },
            elicited: {
                scatter: {
                    xpoints: swebench_scaffolds.map(r => r["release_date"]),
                    ypoints: swebench_scaffolds.map(r => r["SWE-Bench Verified"]),
                    text: swebench_scaffolds.map(r => r["model"]),
                },
                forecast: swebench_forecast_elicited,
            },
            observed_future_data: {
                scatter: {
                    xpoints: swebench_observed_future_data.map(r => r["release_date"]),
                    ypoints: swebench_observed_future_data.map(r => r["SWE-Bench Verified"]),
                    text: swebench_observed_future_data.map(r => r["model"]),
                }
            }
        },
        {
            title: "Cybench",
            yaxis: "Success Rate",
            unelicited: {
                scatter: {
                    xpoints: agentic_benchmark.map(r => r["release_date"]),
                    ypoints: agentic_benchmark.map(r => r["Cybench"]),
                    text: agentic_benchmark.map(r => r["model"]),
                },
                forecast: cybench_forecast,
            },
            elicited: {
                scatter: {
                    xpoints: cybench_scaffolds.map(r => r["release_date"]),
                    ypoints: cybench_scaffolds.map(r => r["Cybench"]),
                    text: cybench_scaffolds.map(r => r["model"]),
                },
                forecast: cybench_forecast_elicited,
            },
        },
        {
            title: "RE-Bench",
            yaxis: "Score",
            unelicited: {
                scatter: {
                    xpoints: agentic_benchmark.map(r => r["release_date"]),
                    ypoints: agentic_benchmark.map(r => r["RE-Bench"]),
                    text: agentic_benchmark.map(r => r["model"]),
                },
                forecast: rebench_forecast,
            },
        }
    ]} />
}


type Scatter = {
    xpoints: Array<number>,
    ypoints: Array<number | null>,
    text: Array<string>,
}

type PlotProps = {
    title: string,
    yaxis: string,
    unelicited: {
        scatter: Scatter,
        forecast: Forecast
    },
    elicited?: {
        scatter: Scatter,
        forecast: Forecast
    }
    observed_future_data?: {
        scatter: Scatter,
    }
}

type InnerPlotProps = {
    plots: Array<PlotProps>
}

function Fig1(props: InnerPlotProps) {
    const [windowWidth, setWindowWidth] = useState(window.innerWidth);

    useEffect(() => {
        const handleResize = () => setWindowWidth(window.innerWidth);
        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    const isMobile = windowWidth < 900;
    const plotWidth = isMobile ? windowWidth - 20 : 1080; // 20px padding for mobile
    const plotHeight = isMobile ? 400 : 500;

    return <Plot
        className=""
        data={
            props.plots.flatMap((plot, i) => {
                const unelicitedScatter = plot.unelicited.scatter;
                const unelicitedForecast = plot.unelicited.forecast;
                const frontier = get_frontier_model_set(
                    unelicitedScatter.xpoints,
                    unelicitedScatter.ypoints,
                    unelicitedScatter.text
                );

                const confidence_interval_data: Plotly.Data[] = [
                    // Add unelicited forecast confidence interval
                    {
                        x: floatYearToDate([...unelicitedForecast.x_linspace, ...unelicitedForecast.x_linspace.slice().reverse()]),
                        y: [...unelicitedForecast.max_ci, ...unelicitedForecast.min_ci.slice().reverse()],
                        fill: 'toself',
                        fillcolor: `${BLUE_RGB}20`,
                        line: { color: 'transparent' },
                        showlegend: false,
                        type: 'scatter',
                        xaxis: `x${i + 1}`,
                        yaxis: `y${i + 1}`,
                        hoverinfo: 'none'
                    }
                ];

                const line_data: Plotly.Data[] = [
                    // Add unelicited forecast line
                    {
                        x: floatYearToDate(unelicitedForecast.x_linspace),
                        y: unelicitedForecast.forecast,
                        mode: 'lines',
                        line: { color: BLUE_RGB },
                        showlegend: false,
                        type: 'scatter',
                        xaxis: `x${i + 1}`,
                        yaxis: `y${i + 1}`,
                        hovertemplate: '%{x}<br>%{y:.3f}<extra></extra>'
                    }
                ];

                const scatter_data: Plotly.Data[] = [
                    // plot frontier points
                    {
                        mode: 'markers',
                        type: 'scatter',
                        x: floatYearToDate(unelicitedScatter.xpoints.filter((_, i) => frontier.has(unelicitedScatter.text[i]))),
                        y: unelicitedScatter.ypoints.filter((_, i) => frontier.has(unelicitedScatter.text[i])),
                        text: unelicitedScatter.text.filter((_, i) => frontier.has(unelicitedScatter.text[i])),
                        xaxis: `x${i + 1}`,
                        yaxis: `y${i + 1}`,
                        marker: {
                            color: BLUE_RGB,
                            symbol: 'star',
                            size: 12
                        },
                        hovertemplate: '%{text}<br>%{y:.3f}<extra></extra>'
                    },
                    // plot non-frontier points
                    {
                        mode: 'markers',
                        type: 'scatter',
                        x: floatYearToDate(unelicitedScatter.xpoints.filter((_, i) => !frontier.has(unelicitedScatter.text[i]))),
                        y: unelicitedScatter.ypoints.filter((_, i) => !frontier.has(unelicitedScatter.text[i])),
                        text: unelicitedScatter.text.filter((_, i) => !frontier.has(unelicitedScatter.text[i])),
                        xaxis: `x${i + 1}`,
                        yaxis: `y${i + 1}`,
                        marker: {
                            color: BLUE_RGB,
                            opacity: 0.5
                        },
                        hovertemplate: '%{text}<br>%{y:.3f}<extra></extra>'
                    }
                ];

                if (plot.elicited) {
                    const elicitedScatter = plot.elicited.scatter;
                    const elicitedForecast = plot.elicited.forecast;
                    const frontier = get_frontier_model_set(
                        elicitedScatter.xpoints,
                        elicitedScatter.ypoints,
                        elicitedScatter.text
                    );

                    confidence_interval_data.push(
                        // Add elicited forecast confidence interval
                        {
                            x: floatYearToDate([...elicitedForecast.x_linspace, ...elicitedForecast.x_linspace.slice().reverse()]),
                            y: [...elicitedForecast.max_ci, ...elicitedForecast.min_ci.slice().reverse()],
                            fill: 'toself',
                            fillcolor: `${ORANGE_RGB}20`,
                            line: { color: 'transparent' },
                            showlegend: false,
                            type: 'scatter',
                            xaxis: `x${i + 1}`,
                            yaxis: `y${i + 1}`,
                            hoverinfo: 'none'
                        }
                    );

                    line_data.push(
                        // Add elicited forecast line
                        {
                            x: floatYearToDate(elicitedForecast.x_linspace),
                            y: elicitedForecast.forecast,
                            mode: 'lines',
                            line: { color: ORANGE_RGB },
                            showlegend: false,
                            type: 'scatter',
                            xaxis: `x${i + 1}`,
                            yaxis: `y${i + 1}`,
                            hovertemplate: '%{x}<br>%{y:.3f}<extra></extra>'
                        }
                    );

                    scatter_data.push(
                        // plot frontier points
                        {
                            mode: 'markers',
                            type: 'scatter',
                            x: floatYearToDate(elicitedScatter.xpoints.filter((_, i) => frontier.has(elicitedScatter.text[i]))),
                            y: elicitedScatter.ypoints.filter((_, i) => frontier.has(elicitedScatter.text[i])),
                            text: elicitedScatter.text,
                            xaxis: `x${i + 1}`,
                            yaxis: `y${i + 1}`,
                            marker: {
                                color: ORANGE_RGB,
                                symbol: 'star',
                                size: 12
                            },
                            hovertemplate: '%{text}<br>%{y:.3f}<extra></extra>'
                        },
                        // plot non-frontier points
                        {
                            mode: 'markers',
                            type: 'scatter',
                            x: floatYearToDate(elicitedScatter.xpoints.filter((_, i) => !frontier.has(elicitedScatter.text[i]))),
                            y: elicitedScatter.ypoints.filter((_, i) => !frontier.has(elicitedScatter.text[i])),
                            text: elicitedScatter.text,
                            xaxis: `x${i + 1}`,
                            yaxis: `y${i + 1}`,
                            marker: {
                                color: ORANGE_RGB,
                                opacity: 0.5
                            },
                            hovertemplate: '%{text}<br>%{y:.3f}<extra></extra>'
                        }
                    );
                }

                // Add observed future data points in red if they exist
                if (plot.observed_future_data) {
                    scatter_data.push({
                        mode: 'markers',
                        type: 'scatter',
                        x: floatYearToDate(plot.observed_future_data.scatter.xpoints),
                        y: plot.observed_future_data.scatter.ypoints,
                        text: plot.observed_future_data.scatter.text,
                        xaxis: `x${i + 1}`,
                        yaxis: `y${i + 1}`,
                        marker: {
                            color: RED_RGB,
                            symbol: 'diamond',
                            size: 8
                        },
                        hovertemplate: '%{text}<br>%{y:.3f}<extra></extra>'
                    });
                }

                return [...confidence_interval_data, ...line_data, ...scatter_data];
            })
        }
        layout={{
            grid: {
                rows: isMobile ? props.plots.length : 1,
                columns: isMobile ? 1 : props.plots.length,
                pattern: 'independent'
            },
            width: plotWidth,
            height: isMobile ? plotHeight * props.plots.length : plotHeight,
            showlegend: false,
            margin: { t: 40, b: 20, l: 40, r: 10 },
            // Add subplot configs
            ...props.plots.reduce((acc, plot, i) => ({
                ...acc,
                [`xaxis${i + 1}`]: {
                    type: 'date',
                    range: floatYearToDate([2022, 2027])
                },
                [`yaxis${i + 1}`]: {
                    range: [
                        Math.min(...plot.unelicited.forecast.y_linspace),
                        Math.max(...plot.unelicited.forecast.y_linspace)
                    ],
                    title: {
                        text: plot.yaxis,
                        font: {
                            size: 16
                        }
                    }
                },
                [`annotations`]: [
                    ...(acc.annotations ?? []),
                    {
                        text: plot.title,
                        showarrow: false,
                        font: {
                            size: 24
                        },
                        xref: `x${i + 1} domain`,
                        yref: `y${i + 1} domain`,
                        x: 0.5,
                        y: 1.05,
                    }
                ]
            }), {} as { [key: string]: any; })
        }}
        config={{ scrollZoom: false, displayModeBar: false, responsive: true }}
    />

}

export default InteractiveFigure1;