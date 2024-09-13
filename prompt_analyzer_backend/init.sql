DO
$do$
BEGIN
   IF NOT EXISTS (
      SELECT
      FROM   pg_catalog.pg_database
      WHERE  datname = 'mydatabase') THEN
      CREATE DATABASE mydatabase;
   END IF;
END
$do$;

CREATE EXTENSION IF NOT EXISTS timescaledb;