"""The eval corpus.

Ground truth is the reference query, not a frozen number. theLook is appended to
continuously — its newest order is dated today — so a literal expected value
starts rotting the day it is written. The seed trio for "loyal" already carries
this scar: its report says 5,746, which was true when it was written and drifts
a little every day. Re-executing the reference query means the truth moves with
the data and a failure always means the agent was wrong.

Reference SQL for the definition-dependent cases is copied from the Golden
Bucket trios deliberately. The question under test is not "can the agent invent
a definition of loyal" — there is no right answer to that — but "given the
definition, does it compute it correctly". Four live runs produced 0, 1254 and
66133 for the same question, which is what this exists to catch.

The tables are qualified by the guard at check time, so bare names are fine and
match what the agent itself produces.
"""

from __future__ import annotations

from retail_agent.evals.types import EvalCase

DATASET = "`bigquery-public-data.thelook_ecommerce"

# Completed orders only. Every revenue and count figure in this corpus excludes
# Cancelled and Returned, matching the trios, because mixing the two conventions
# is the single easiest way to produce two defensible numbers for one question.
COMPLETED = "status NOT IN ('Cancelled', 'Returned')"


EVAL_CASES: tuple[EvalCase, ...] = (
    # --- definition-dependent: the cases that motivated the suite ---
    EvalCase(
        id="loyal-count",
        question="How many loyal customers do we have?",
        reference_sql=f"""
            SELECT COUNT(*) AS loyal_customers FROM (
              SELECT o.user_id
              FROM {DATASET}.orders` AS o
              WHERE o.{COMPLETED}
              GROUP BY o.user_id
              HAVING COUNT(DISTINCT o.order_id) >= 3
            )
        """,
        required_definitions=("loyal",),
        notes="Four live runs returned 0, 1254 and 66133 for this question.",
    ),
    EvalCase(
        id="loyal-share",
        question="What share of our customers are loyal?",
        reference_sql=f"""
            WITH per_customer AS (
              SELECT user_id, COUNT(DISTINCT order_id) AS orders
              FROM {DATASET}.orders`
              WHERE {COMPLETED}
              GROUP BY user_id
            )
            SELECT ROUND(COUNTIF(orders >= 3) / COUNT(*) * 100, 2) AS pct_loyal
            FROM per_customer
        """,
        required_definitions=("loyal",),
        tolerance=0.02,
    ),
    EvalCase(
        id="engaged-count",
        question="How many engaged customers do we have?",
        reference_sql=f"""
            SELECT COUNT(*) AS engaged FROM (
              SELECT user_id
              FROM {DATASET}.orders`
              WHERE {COMPLETED}
                AND created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
              GROUP BY user_id
              HAVING COUNT(DISTINCT order_id) >= 2
            )
        """,
        required_definitions=("engaged",),
        notes="Recency measure, unlike loyal. Confusing the two is the failure.",
    ),
    EvalCase(
        id="churn-count",
        question="How many customers have churned?",
        reference_sql=f"""
            WITH active_before AS (
              SELECT DISTINCT user_id FROM {DATASET}.orders`
              WHERE {COMPLETED}
                AND created_at BETWEEN TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 270 DAY)
                                   AND TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
            ),
            active_recently AS (
              SELECT DISTINCT user_id FROM {DATASET}.orders`
              WHERE {COMPLETED}
                AND created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
            )
            SELECT COUNT(*) AS churned
            FROM active_before
            WHERE user_id NOT IN (SELECT user_id FROM active_recently)
        """,
        required_definitions=("churn",),
    ),
    EvalCase(
        id="top-customers-ranked",
        question="Who are our top 10 customers by lifetime spend?",
        reference_sql=f"""
            SELECT o.user_id
            FROM {DATASET}.order_items` AS oi
            JOIN {DATASET}.orders` AS o ON o.order_id = oi.order_id
            WHERE o.{COMPLETED}
            GROUP BY o.user_id
            ORDER BY SUM(oi.sale_price) DESC, o.user_id
            LIMIT 10
        """,
        required_definitions=("top",),
        ranked=True,
        notes="Ordering is the answer; the right ten in the wrong order is wrong.",
    ),
    EvalCase(
        id="top-customer-spend",
        question="How much has our single biggest customer spent with us?",
        reference_sql=f"""
            SELECT ROUND(SUM(oi.sale_price), 2) AS spend
            FROM {DATASET}.order_items` AS oi
            JOIN {DATASET}.orders` AS o ON o.order_id = oi.order_id
            WHERE o.{COMPLETED}
            GROUP BY o.user_id
            ORDER BY spend DESC
            LIMIT 1
        """,
        required_definitions=("top",),
        tolerance=0.001,
    ),
    EvalCase(
        id="underspending-states",
        question="Which states are underspending?",
        reference_sql=f"""
            WITH per_customer AS (
              SELECT u.state, u.id AS user_id, SUM(oi.sale_price) AS total
              FROM {DATASET}.users` AS u
              JOIN {DATASET}.orders` AS o ON o.user_id = u.id
              JOIN {DATASET}.order_items` AS oi ON oi.order_id = o.order_id
              WHERE o.{COMPLETED}
              GROUP BY u.state, u.id
            ),
            per_state AS (
              SELECT state, AVG(total) AS avg_spend FROM per_customer GROUP BY state
            )
            SELECT COUNT(*) AS underspending_states
            FROM per_state
            WHERE avg_spend < (SELECT AVG(avg_spend) * 0.9 FROM per_state)
        """,
        required_definitions=("underspending",),
        notes="Per customer, not per state total — the trio says so explicitly.",
    ),
    EvalCase(
        id="brands-performing-well",
        question="How many brands are performing well?",
        reference_sql=f"""
            WITH brand AS (
              SELECT p.brand,
                     SUM(oi.sale_price) AS revenue,
                     SUM(oi.sale_price - p.cost) / NULLIF(SUM(oi.sale_price), 0) AS margin
              FROM {DATASET}.order_items` AS oi
              JOIN {DATASET}.orders` AS o ON o.order_id = oi.order_id
              JOIN {DATASET}.products` AS p ON p.id = oi.product_id
              WHERE o.{COMPLETED} AND p.brand IS NOT NULL
              GROUP BY p.brand
            ),
            cuts AS (
              SELECT APPROX_QUANTILES(revenue, 2)[OFFSET(1)] AS med_revenue,
                     APPROX_QUANTILES(margin, 2)[OFFSET(1)] AS med_margin
              FROM brand
            )
            SELECT COUNT(*) AS performing
            FROM brand, cuts
            WHERE revenue > med_revenue AND margin > med_margin
        """,
        required_definitions=("performing well",),
        notes="Both revenue AND margin above median. Revenue alone is the trap.",
    ),
    # --- plain aggregates: the agent should never get these wrong ---
    EvalCase(
        id="total-users",
        question="How many users are in the database?",
        reference_sql=f"SELECT COUNT(*) AS n FROM {DATASET}.users`",
    ),
    EvalCase(
        id="total-orders",
        question="How many orders have been placed in total?",
        reference_sql=f"SELECT COUNT(*) AS n FROM {DATASET}.orders`",
    ),
    EvalCase(
        id="completed-orders",
        question="How many orders were completed, excluding cancellations and returns?",
        reference_sql=f"SELECT COUNT(*) AS n FROM {DATASET}.orders` WHERE {COMPLETED}",
    ),
    EvalCase(
        id="total-products",
        question="How many distinct products do we sell?",
        reference_sql=f"SELECT COUNT(*) AS n FROM {DATASET}.products`",
    ),
    EvalCase(
        id="distinct-brands",
        question="How many different brands do we carry?",
        reference_sql=f"SELECT COUNT(DISTINCT brand) AS n FROM {DATASET}.products`",
    ),
    EvalCase(
        id="distinct-categories",
        question="How many product categories are there?",
        reference_sql=f"SELECT COUNT(DISTINCT category) AS n FROM {DATASET}.products`",
    ),
    EvalCase(
        id="lifetime-revenue",
        question="What is our total revenue from completed orders, all time?",
        reference_sql=f"""
            SELECT ROUND(SUM(oi.sale_price), 2) AS revenue
            FROM {DATASET}.order_items` AS oi
            JOIN {DATASET}.orders` AS o ON o.order_id = oi.order_id
            WHERE o.{COMPLETED}
        """,
        tolerance=0.001,
    ),
    EvalCase(
        id="average-order-value",
        question="What is the average order value for completed orders?",
        reference_sql=f"""
            WITH per_order AS (
              SELECT o.order_id, SUM(oi.sale_price) AS total
              FROM {DATASET}.orders` AS o
              JOIN {DATASET}.order_items` AS oi ON oi.order_id = o.order_id
              WHERE o.{COMPLETED}
              GROUP BY o.order_id
            )
            SELECT ROUND(AVG(total), 2) AS aov FROM per_order
        """,
        tolerance=0.001,
        notes="Per order, not per order_item. Averaging line items is the bug.",
    ),
    EvalCase(
        id="customers-who-ordered",
        question="How many customers have ever placed a completed order?",
        reference_sql=f"""
            SELECT COUNT(DISTINCT user_id) AS n FROM {DATASET}.orders` WHERE {COMPLETED}
        """,
    ),
    EvalCase(
        id="one-order-customers",
        question="How many customers have placed exactly one completed order?",
        reference_sql=f"""
            SELECT COUNT(*) AS n FROM (
              SELECT user_id FROM {DATASET}.orders` WHERE {COMPLETED}
              GROUP BY user_id HAVING COUNT(DISTINCT order_id) = 1
            )
        """,
    ),
    EvalCase(
        id="cancelled-orders",
        question="How many orders were cancelled?",
        reference_sql=f"SELECT COUNT(*) AS n FROM {DATASET}.orders` WHERE status = 'Cancelled'",
    ),
    EvalCase(
        id="returned-orders",
        question="How many orders were returned?",
        reference_sql=f"SELECT COUNT(*) AS n FROM {DATASET}.orders` WHERE status = 'Returned'",
    ),
    EvalCase(
        id="return-rate",
        question="What percentage of all orders were returned?",
        reference_sql=f"""
            SELECT ROUND(COUNTIF(status = 'Returned') / COUNT(*) * 100, 3) AS pct
            FROM {DATASET}.orders`
        """,
        tolerance=0.02,
    ),
    EvalCase(
        id="items-per-order",
        question="On average, how many items are in a completed order?",
        reference_sql=f"""
            SELECT ROUND(COUNT(*) / COUNT(DISTINCT o.order_id), 4) AS items
            FROM {DATASET}.orders` AS o
            JOIN {DATASET}.order_items` AS oi ON oi.order_id = o.order_id
            WHERE o.{COMPLETED}
        """,
        tolerance=0.01,
    ),
    # --- segmentation ---
    EvalCase(
        id="female-users",
        question="How many of our users are women?",
        reference_sql=f"SELECT COUNT(*) AS n FROM {DATASET}.users` WHERE gender = 'F'",
    ),
    EvalCase(
        id="distinct-countries",
        question="How many countries do our customers come from?",
        reference_sql=f"SELECT COUNT(DISTINCT country) AS n FROM {DATASET}.users`",
    ),
    EvalCase(
        id="top-country-by-users",
        question="Which country has the most users?",
        reference_sql=f"""
            SELECT country FROM {DATASET}.users`
            GROUP BY country ORDER BY COUNT(*) DESC, country LIMIT 1
        """,
        answer_column="country",
        notes=(
            "Originally asked '...and how many?' while the reference returned "
            "only the count, so a correct 'China' scored wrong. A two-part "
            "question needs a two-part reference or neither part."
        ),
    ),
    EvalCase(
        id="top-5-countries-ranked",
        question="What are the top 5 countries by number of users?",
        reference_sql=f"""
            SELECT country FROM {DATASET}.users`
            GROUP BY country ORDER BY COUNT(*) DESC, country LIMIT 5
        """,
        ranked=True,
    ),
    EvalCase(
        id="average-user-age",
        question="What is the average age of our users?",
        reference_sql=f"SELECT ROUND(AVG(age), 4) AS age FROM {DATASET}.users`",
        tolerance=0.01,
    ),
    EvalCase(
        id="traffic-source-count",
        question="How many users came from Search as their traffic source?",
        reference_sql=f"""
            SELECT COUNT(*) AS n FROM {DATASET}.users` WHERE traffic_source = 'Search'
        """,
    ),
    EvalCase(
        id="top-3-traffic-sources-ranked",
        question="What are the three biggest traffic sources by user count?",
        reference_sql=f"""
            SELECT traffic_source FROM {DATASET}.users`
            GROUP BY traffic_source ORDER BY COUNT(*) DESC, traffic_source LIMIT 3
        """,
        ranked=True,
    ),
    # --- product and brand ---
    EvalCase(
        id="top-brand-by-revenue",
        question="Which brand has the highest revenue from completed orders?",
        reference_sql=f"""
            SELECT p.brand
            FROM {DATASET}.order_items` AS oi
            JOIN {DATASET}.orders` AS o ON o.order_id = oi.order_id
            JOIN {DATASET}.products` AS p ON p.id = oi.product_id
            WHERE o.{COMPLETED}
            GROUP BY p.brand
            ORDER BY SUM(oi.sale_price) DESC, p.brand
            LIMIT 1
        """,
        answer_column="brand",
    ),
    EvalCase(
        id="top-5-brands-ranked",
        question="Rank our top 5 brands by revenue from completed orders.",
        reference_sql=f"""
            SELECT p.brand
            FROM {DATASET}.order_items` AS oi
            JOIN {DATASET}.orders` AS o ON o.order_id = oi.order_id
            JOIN {DATASET}.products` AS p ON p.id = oi.product_id
            WHERE o.{COMPLETED}
            GROUP BY p.brand
            ORDER BY SUM(oi.sale_price) DESC, p.brand
            LIMIT 5
        """,
        ranked=True,
    ),
    EvalCase(
        id="top-category-by-revenue",
        question="Which product category earns us the most revenue?",
        reference_sql=f"""
            SELECT p.category
            FROM {DATASET}.order_items` AS oi
            JOIN {DATASET}.orders` AS o ON o.order_id = oi.order_id
            JOIN {DATASET}.products` AS p ON p.id = oi.product_id
            WHERE o.{COMPLETED}
            GROUP BY p.category
            ORDER BY SUM(oi.sale_price) DESC, p.category
            LIMIT 1
        """,
        answer_column="category",
    ),
    EvalCase(
        id="gross-margin-pct",
        question="What is our overall gross margin percentage on completed orders?",
        reference_sql=f"""
            SELECT ROUND(SUM(oi.sale_price - p.cost) / SUM(oi.sale_price) * 100, 3) AS margin
            FROM {DATASET}.order_items` AS oi
            JOIN {DATASET}.orders` AS o ON o.order_id = oi.order_id
            JOIN {DATASET}.products` AS p ON p.id = oi.product_id
            WHERE o.{COMPLETED}
        """,
        tolerance=0.02,
    ),
    EvalCase(
        id="most-expensive-product",
        question="What is the retail price of our most expensive product?",
        reference_sql=f"SELECT MAX(retail_price) AS price FROM {DATASET}.products`",
        tolerance=0.001,
    ),
    EvalCase(
        id="products-never-sold",
        question=(
            "How many products have never appeared in any order item at all, "
            "counting cancelled and returned ones too?"
        ),
        reference_sql=f"""
            SELECT COUNT(*) AS n FROM {DATASET}.products` AS p
            WHERE NOT EXISTS (
              SELECT 1 FROM {DATASET}.order_items` AS oi WHERE oi.product_id = p.id
            )
        """,
    ),
    EvalCase(
        id="distinct-departments",
        question="How many departments does our catalogue have?",
        reference_sql=f"SELECT COUNT(DISTINCT department) AS n FROM {DATASET}.products`",
    ),
    # --- time windows: where an off-by-one boundary hides ---
    EvalCase(
        id="orders-last-30-days",
        question="How many orders were placed in the last 30 days?",
        reference_sql=f"""
            SELECT COUNT(*) AS n FROM {DATASET}.orders`
            WHERE created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
        """,
        tolerance=0.02,
        notes="Tolerant: the dataset is appended to while the suite runs.",
    ),
    EvalCase(
        id="orders-in-2023",
        question="How many orders were placed during 2023?",
        reference_sql=f"""
            SELECT COUNT(*) AS n FROM {DATASET}.orders`
            WHERE created_at >= TIMESTAMP('2023-01-01')
              AND created_at < TIMESTAMP('2024-01-01')
        """,
        notes="A closed historical window, so this one genuinely cannot drift.",
    ),
    EvalCase(
        id="revenue-in-2023",
        question="What was our revenue from completed orders in 2023?",
        reference_sql=f"""
            SELECT ROUND(SUM(oi.sale_price), 2) AS revenue
            FROM {DATASET}.order_items` AS oi
            JOIN {DATASET}.orders` AS o ON o.order_id = oi.order_id
            WHERE o.{COMPLETED}
              AND o.created_at >= TIMESTAMP('2023-01-01')
              AND o.created_at < TIMESTAMP('2024-01-01')
        """,
        tolerance=0.001,
    ),
    EvalCase(
        id="busiest-month-2023",
        question=(
            "Which calendar month of 2023 had the most orders? "
            "Answer with the month number, 1 to 12."
        ),
        reference_sql=f"""
            SELECT EXTRACT(MONTH FROM created_at) AS month
            FROM {DATASET}.orders`
            WHERE created_at >= TIMESTAMP('2023-01-01')
              AND created_at < TIMESTAMP('2024-01-01')
            GROUP BY month ORDER BY COUNT(*) DESC, month LIMIT 1
        """,
        answer_column="month",
        notes=(
            "The agent returns (year, month, total_orders), so the column is "
            "named. It also answered '2023-12' once, which is correct but "
            "unscoreable — hence the question pins the format."
        ),
    ),
    EvalCase(
        id="first-order-year",
        question="What year was the earliest order in the dataset placed?",
        reference_sql=f"SELECT EXTRACT(YEAR FROM MIN(created_at)) AS y FROM {DATASET}.orders`",
    ),
    EvalCase(
        id="new-customers-2023",
        question="How many customers placed their first ever order in 2023?",
        reference_sql=f"""
            WITH first_order AS (
              SELECT user_id, MIN(created_at) AS first_at
              FROM {DATASET}.orders`
              GROUP BY user_id
            )
            SELECT COUNT(*) AS n FROM first_order
            WHERE first_at >= TIMESTAMP('2023-01-01') AND first_at < TIMESTAMP('2024-01-01')
        """,
    ),
    EvalCase(
        id="repeat-rate-2023",
        question="What share of customers ordering in 2023 ordered more than once that year?",
        reference_sql=f"""
            WITH per_customer AS (
              SELECT user_id, COUNT(DISTINCT order_id) AS orders
              FROM {DATASET}.orders`
              WHERE {COMPLETED}
                AND created_at >= TIMESTAMP('2023-01-01')
                AND created_at < TIMESTAMP('2024-01-01')
              GROUP BY user_id
            )
            SELECT ROUND(COUNTIF(orders > 1) / COUNT(*) * 100, 3) AS pct FROM per_customer
        """,
        tolerance=0.02,
    ),
    # --- shape of the data, where a wrong join multiplies rows ---
    EvalCase(
        id="order-items-total",
        question="How many order line items are there in total?",
        reference_sql=f"SELECT COUNT(*) AS n FROM {DATASET}.order_items`",
    ),
    EvalCase(
        id="distinct-order-statuses",
        question="How many distinct order statuses exist?",
        reference_sql=f"SELECT COUNT(DISTINCT status) AS n FROM {DATASET}.orders`",
    ),
    EvalCase(
        id="max-orders-by-one-customer",
        question="What is the highest number of completed orders placed by a single customer?",
        reference_sql=f"""
            SELECT MAX(orders) AS n FROM (
              SELECT COUNT(DISTINCT order_id) AS orders
              FROM {DATASET}.orders` WHERE {COMPLETED}
              GROUP BY user_id
            )
        """,
    ),
    EvalCase(
        id="distribution-centre-count",
        question="How many distribution centres supply our products?",
        reference_sql=f"""
            SELECT COUNT(DISTINCT distribution_center_id) AS n FROM {DATASET}.products`
        """,
    ),
)
