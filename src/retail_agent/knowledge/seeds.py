"""The seed corpus.

The brief marks the Golden Bucket theoretical, so these are hand-authored
rather than exported from a real warehouse. They are written the way a real
corpus would be: the definitions encode judgements the schema cannot settle,
and each one is a decision somebody would defend in a meeting.

theLook has no subscriptions and no cancellations, so churn genuinely cannot be
read off the columns. The definition below is a choice — 180 days of history,
90 days of silence — and the point of storing it is that the agent uses *that*
choice consistently instead of inventing a new one per question.
"""

from __future__ import annotations

from retail_agent.knowledge.trios import Trio

SEED_TRIOS: tuple[Trio, ...] = (
    Trio(
        id="churn-90",
        question="Why did our churn rate spike last month?",
        sql=(
            "WITH active_before AS (\n"
            "  SELECT DISTINCT user_id FROM `bigquery-public-data.thelook_ecommerce.orders`\n"
            "  WHERE created_at BETWEEN TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 270 DAY)\n"
            "                       AND TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)\n"
            "    AND status NOT IN ('Cancelled', 'Returned')\n"
            "), ordered_since AS (\n"
            "  SELECT DISTINCT user_id FROM `bigquery-public-data.thelook_ecommerce.orders`\n"
            "  WHERE created_at > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)\n"
            "    AND status NOT IN ('Cancelled', 'Returned')\n"
            ")\n"
            "SELECT COUNT(*) AS churned FROM active_before\n"
            "WHERE user_id NOT IN (SELECT user_id FROM ordered_since)"
        ),
        report=(
            "Churn reached 4.1% in the quarter, up from 3.4%.\n\n"
            "The rise is concentrated in customers acquired through Email in "
            "2023: they churn at 6.2% against 3.1% for Organic, and they are "
            "35% of the cohort. Excluding them, churn is flat.\n\n"
            "1. Review the Email acquisition offer — discount-led sign-ups are "
            "not repeating.\n"
            "2. Target the 1,240 customers at 60-89 days silent before they "
            "cross the threshold.\n"
            "3. Re-check in 30 days; one quarter is not a trend."
        ),
        metric_definitions={
            "churn": (
                "ordered at least once between 270 and 90 days ago, and nothing "
                "in the trailing 90 days. Excludes Cancelled and Returned orders. "
                "theLook has no subscriptions, so this is a behavioural "
                "definition, not a contractual one"
            ),
            "churned": (
                "a customer meeting the churn definition: active before the "
                "window, silent for 90 days"
            ),
        },
        tags=("churn", "retention", "cohort"),
        author="analytics",
        version=1,
    ),
    Trio(
        id="top-customers",
        question="Who are our top customers?",
        sql=(
            "SELECT o.user_id, ROUND(SUM(oi.sale_price), 2) AS lifetime_spend\n"
            "FROM `bigquery-public-data.thelook_ecommerce.order_items` AS oi\n"
            "JOIN `bigquery-public-data.thelook_ecommerce.orders` AS o\n"
            "  ON oi.order_id = o.order_id\n"
            "WHERE oi.status NOT IN ('Cancelled', 'Returned')\n"
            "GROUP BY o.user_id ORDER BY lifetime_spend DESC LIMIT 10"
        ),
        report=(
            "The top 10 customers by lifetime spend account for $13,707, an "
            "average of $1,371 each — roughly 18x the median customer.\n\n"
            "Concentration is low: no single customer is above 0.1% of revenue, "
            "so there is no account whose loss would show up in the monthly "
            "numbers.\n\n"
            "1. Treat this as a segment, not as accounts to manage "
            "individually.\n"
            "2. Compare their category mix against the median customer before "
            "building any retention offer."
        ),
        metric_definitions={
            "top": (
                "ranked by lifetime spend — SUM(order_items.sale_price) across "
                "all time, excluding Cancelled and Returned. Default to 10 "
                "unless the question says otherwise"
            ),
            "high value": (
                "lifetime spend in the top decile. Not the same as top: 'top' "
                "is a ranking, 'high value' is a threshold"
            ),
        },
        tags=("customers", "spend", "ranking", "revenue"),
        author="analytics",
        version=1,
    ),
    Trio(
        id="underspending",
        question="Why are users in state X underspending?",
        sql=(
            "SELECT u.state, ROUND(AVG(spend.total), 2) AS avg_customer_spend\n"
            "FROM `bigquery-public-data.thelook_ecommerce.users` AS u\n"
            "JOIN (\n"
            "  SELECT o.user_id, SUM(oi.sale_price) AS total\n"
            "  FROM `bigquery-public-data.thelook_ecommerce.order_items` AS oi\n"
            "  JOIN `bigquery-public-data.thelook_ecommerce.orders` AS o\n"
            "    ON oi.order_id = o.order_id\n"
            "  WHERE oi.status NOT IN ('Cancelled', 'Returned')\n"
            "  GROUP BY o.user_id\n"
            ") AS spend ON spend.user_id = u.id\n"
            "GROUP BY u.state ORDER BY avg_customer_spend"
        ),
        report=(
            "Average spend per customer in the state is $94 against a national "
            "average of $118 — 20% below, and the gap is in order frequency "
            "rather than basket size.\n\n"
            "Basket size is within $3 of the national figure. Customers there "
            "order 1.4 times on average against 1.8 nationally.\n\n"
            "1. The problem is repeat purchase, not pricing — do not discount.\n"
            "2. Check delivery times for the state's distribution centre "
            "against the national median."
        ),
        metric_definitions={
            "underspending": (
                "average spend per customer at least 10% below the all-state "
                "average for the same period. Always compare per customer, not "
                "per state total, or population size dominates the answer"
            ),
            "overspending": (
                "average spend per customer at least 10% above the all-state "
                "average for the same period"
            ),
        },
        tags=("spend", "state", "geography", "comparison", "underspending"),
        author="analytics",
        version=1,
    ),
    Trio(
        id="loyal-customers",
        question="How many loyal customers do we have?",
        sql=(
            "SELECT COUNT(*) AS loyal_customers FROM (\n"
            "  SELECT o.user_id\n"
            "  FROM `bigquery-public-data.thelook_ecommerce.orders` AS o\n"
            "  WHERE o.status NOT IN ('Cancelled', 'Returned')\n"
            "  GROUP BY o.user_id\n"
            "  HAVING COUNT(DISTINCT o.order_id) >= 3\n"
            ")"
        ),
        report=(
            "5,746 customers have placed three or more completed orders — 5.7% "
            "of everyone who has ordered at all.\n\n"
            "The distribution is steep: 71% of customers have ordered exactly "
            "once, and the three-order threshold is where repeat behaviour "
            "starts to look deliberate rather than incidental.\n\n"
            "1. Treat the one-order majority as the acquisition problem it is.\n"
            "2. Measure loyalty movement quarterly at this threshold rather "
            "than re-cutting it each time it is asked."
        ),
        metric_definitions={
            "loyal": (
                "three or more completed orders, all time, counting distinct "
                "orders and excluding Cancelled and Returned. Count orders, not "
                "order_items — joining to order_items multiplies rows per order "
                "and inflates the count"
            ),
            "engaged": (
                "at least two completed orders in the trailing 180 days. Unlike "
                "loyal, this is a recency measure, not a lifetime one"
            ),
        },
        tags=("loyal", "engaged", "customers", "retention", "orders"),
        author="analytics",
        version=1,
    ),
    Trio(
        id="brand-performance",
        question="Which brands are performing well?",
        sql=(
            "SELECT p.brand, ROUND(SUM(oi.sale_price), 2) AS revenue,\n"
            "       COUNT(*) AS units,\n"
            "       ROUND(SUM(oi.sale_price) - SUM(p.cost), 2) AS margin\n"
            "FROM `bigquery-public-data.thelook_ecommerce.order_items` AS oi\n"
            "JOIN `bigquery-public-data.thelook_ecommerce.products` AS p\n"
            "  ON oi.product_id = p.id\n"
            "WHERE oi.status NOT IN ('Cancelled', 'Returned')\n"
            "GROUP BY p.brand ORDER BY revenue DESC LIMIT 20"
        ),
        report=(
            "The top 20 brands by revenue carry 31% of sales. Calvin Klein "
            "leads on revenue; margin tells a different story, where it ranks "
            "seventh.\n\n"
            "Revenue rank and margin rank disagree for 11 of the 20. Ranking on "
            "revenue alone would put effort behind brands that sell volume at "
            "thin margin.\n\n"
            "1. Rank on margin for any inventory or promotion decision.\n"
            "2. Investigate the four brands in the top ten on both."
        ),
        metric_definitions={
            "performing well": (
                "above the median on BOTH revenue and margin for the period. "
                "Revenue alone rewards volume at thin margin"
            ),
            "underperforming": (
                "below the median on both revenue and margin for the period"
            ),
            "margin": (
                "SUM(order_items.sale_price) - SUM(products.cost) for the same "
                "rows. theLook has no discount column, so sale_price is what "
                "was actually charged"
            ),
        },
        tags=("brand", "product", "revenue", "margin", "performance"),
        author="analytics",
        version=1,
    ),
)
