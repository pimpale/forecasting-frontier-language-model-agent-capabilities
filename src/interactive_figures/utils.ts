export const BLUE_RGB = "#1f77b4";
export const ORANGE_RGB = "#ff7f0e";


export type Forecast = {
    x_linspace: Array<number>,
    y_linspace: Array<number>,
    forecast: Array<number>,
    min_ci: Array<number>,
    max_ci: Array<number>,
    density: Array<Array<number>>,
}

export function floatYearToDate(years: number[]): number[] {
    return years.map(year => {
        const wholeYear = Math.floor(year);
        const fraction = year - wholeYear;
        const millisInYear = 365.25 * 24 * 60 * 60 * 1000; // Account for leap years
        const startOfYear = new Date(wholeYear, 0, 1).getTime();
        return startOfYear + (fraction * millisInYear);
    });
}

export function get_frontier_model_set(
    x: Array<number>,
    y: Array<number | null>,
    model: Array<string>,
): Set<string> {
    // sort by x
    const joint = x.map((v, i) => [v, y[i], model[i]] as [number, number | null, string])
        // first sort by y, then by x
        .sort((a, b) => (b[1] ?? 0) - (a[1] ?? 0))
        .sort((a, b) => a[0] - b[0]);
    let frontier: Set<string> = new Set();
    let max = -Infinity;
    for (let [_, y, m] of joint) {
        if (y === null) {
            continue;
        }
        if (y > max) {
            frontier.add(m);
            max = y;
        }
    }
    return frontier;
}
