import csv

from common import BUILD, DATA, write_rows

TABLES = {
    1: ("validation/accuracy.csv", ("method", "tau", "error", "order", "loop_seconds")),
    2: ("validation/refinement.csv", ("d", "n", "tau", "difference", "ratio_order")),
    3: (
        "joint_accuracy_cost/summary.csv",
        ("mode", "tau", "density", "gradient", "iterations", "seconds"),
    ),
    4: (
        "residual_accumulation/summary.csv",
        ("tau", "error_beta3", "order_beta3", "error_beta2", "order_beta2"),
    ),
}


def export_table(number, source, output):
    fields = TABLES[number][1]
    with source.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if number == 4:
        by_case = {(row["tau"], row["power"]): row for row in rows}
        rows = [
            {
                "tau": tau,
                "error_beta3": by_case[tau, "3"]["maximum_density_difference"],
                "order_beta3": by_case[tau, "3"]["order"],
                "error_beta2": by_case[tau, "2"]["maximum_density_difference"],
                "order_beta2": by_case[tau, "2"]["order"],
            }
            for tau in sorted({row["tau"] for row in rows}, key=float, reverse=True)
        ]
    rows = [
        {key: "" if row[key] == "nan" else row[key] for key in fields} for row in rows
    ]
    write_rows(output / f"table{number}.csv", rows)


if __name__ == "__main__":
    for number, (source, _) in TABLES.items():
        export_table(number, DATA / source, BUILD / "archive")
