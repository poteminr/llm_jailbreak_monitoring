To run locally use two commands:

```
make build
```
and after it
```
make run
```

after it go to swagger with all endpoints of API

```
localhost:8000/docs#
```

# Grafana metrics

```
-- Количество инъекционных промптов по времени --

SELECT
  date_trunc('hour', timestamp) AS time,
  COUNT(*) as value
FROM prompts
WHERE categorised_as = 'injection'
GROUP BY time
ORDER BY time;
```

```
-- Количество unsafe генераций по времени --

SELECT
  date_trunc('hour', timestamp) AS time,
  COUNT(*) as value
FROM prompts
WHERE categorised_as = 'unsafe'
GROUP BY time
ORDER BY time;
```

```
-- Отношение unsafe генераций к инъекционным (unsafe/injection) --

SELECT
  date_trunc('hour', timestamp) AS time,
  (COUNT(CASE WHEN categorised_as = 'unsafe' THEN 1 END)::float / NULLIF(COUNT(CASE WHEN categorised_as = 'injection' THEN 1 END), 0)) as ratio
FROM prompts
GROUP BY time
ORDER BY time;
```

```
-- Средняя длина джаилбрейк-промптов --

SELECT
  date_trunc('hour', timestamp) AS time,
  AVG(LENGTH(prompt)) as avg_length
FROM prompts
WHERE categorised_as = 'jailbreak'
GROUP BY time
ORDER BY time;
```

```
-- M (Доля инъекционных промптов) --

SELECT
  date_trunc('hour', timestamp) AS time,
  (COUNT(CASE WHEN categorised_as = 'injection' THEN 1 END)::float / COUNT(*)) as M_ratio
FROM prompts
GROUP BY time
ORDER BY time;
```

```
-- N (Доля unsafe генераций) --

SELECT
  date_trunc('hour', timestamp) AS time,
  (COUNT(CASE WHEN categorised_as = 'unsafe' THEN 1 END)::float / COUNT(*)) as N_ratio
FROM prompts
GROUP BY time
ORDER BY time;
```

```
-- Частотность инъекционных промптов (повторяющиеся инъекционные промпты) --

SELECT
  prompt,
  COUNT(*) as frequency
FROM prompts
WHERE categorised_as = 'injection'
GROUP BY prompt
HAVING COUNT(*) > 1
ORDER BY frequency DESC;
```
