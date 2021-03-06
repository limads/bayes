/*
This files contain SQL code (compatible with PostgreSQL) to manipulate the
bayes JSON probability distribution specification. It might be useful if you
are storing your models in a database as a JSON field. Using those routines,
you will be able to:
- Sample from probabilistic models
- Access probability distirbution parameters
For actual inference, it is best to use the bayes CLI. Use those routines to
make predictions after you inference step is over.
*/

/*-- # Model building

-- Creates a bivariate normal (correlated)
-- select bivariate(normal(v1, 10, 2), normal(v2, 10, 4), 0.3);
create function bivariate(a text, b text, cor double precision) returns text as $$

$$ language sql;

-- Creates a multivariate normal distribution from continuous variables assumed un-correlated
create function joint(variadic data double precision) returns text as $$
    select array_agg(data);
$$ language sql;

-- Creates a multivariate normal from a pair of bivariates, conditioning the correlations
-- to the informed higher-order correlation.
create function joint(biv text, new text, cor double precision) returns text as $$

$$ language sql;

-- Conditions distribution a over distribution b, such
-- that the resulting distribution now factors as p(a|b)p(b)
create function condition(a text, b text) returns text as $$

$$ language sql;*/

-- # General-purpose distribution evaluation and sampling

-- Note: $libdir is actually pkglibdir of pg_config, NOT libdir

create or replace function observe_data(tbl text) returns json as $$
declare
    out json;
begin
    execute format('select array_to_json(array_agg(row_to_json(p))) from %I p;', tbl) into out;
    return out;
end
$$ language plpgsql;

create function log_prob(distr text) returns double precision as
    '$libdir/libbayes.so', 'log_prob'
language c strict;

-- select * from sample_from('{ "prop":0.5, "n":10 }');
create function sample_from(distr text) returns text as
    '$libdir/libbayes.so', 'sample_from'
language c strict;

/*create function sample(distr text) returns setof double precision as $$
    select cast(
        cast(json_array_elements(sample_from(distr))::json as text) as double precision
    )
$$ language sql;*/

-- # Specific distribtions
-- ## Bernoulli
-- Create with:
-- select row(array['t', 'f'], 0.2)::bernoulli;
create type bernoulli as (
    data boolean[],
    n integer,
    prop real
);

create type distribution as (
    data real[],
    n integer,
    loc real,
    scale real
);

-- Conditioning function aggregates
-- Supports the syntax condition(a, b) or a|b
create function condition(a distribution, b distribution) returns distribution as $$
    select a;
$$ language sql;

-- Supports the syntax condition(array[1.0,2.0], a) or array[1.0,2.0] | a
create function condition(a real[], b distribution) returns distribution as $$
    select b;
$$ language sql;

create operator | (
    procedure = condition,
    leftarg = distribution,
    rightarg = distribution
);

create operator | (
    procedure = condition,
    leftarg = real[],
    rightarg = distribution
);

create function append_distr(distribution, real) returns distribution as $$
    select row(array_append(($1).data, $2), ($1).n + 1, ($1).loc, ($1).scale)::distribution;
$$ language sql;

create aggregate condition(real, distribution) (
    sfunc = append_distr,
    stype = (real[], distribution),
    finalfunc = condition
);

-- Normal distribution functions

create function norm(real[]) returns distribution as $$
    select row($1, array_length($1, 1), avg(d), variance(d))::distribution
    from unnest($1) d;
$$ language sql;

create function norm(real, real) returns distribution as $$
    select row(NULL, 1, $1, $2)::distribution; 
$$ language sql;

create aggregate norm(real) (
    sfunc = array_append,
    stype = real[],
    finalfunc = norm
);

-- We can define the method "sample" as the one that builds a JSON representation
-- for the informed table from the catalog. Then, we condition this sample over the full
-- probabilistic model
-- select posterior(sample('patients') | model );

-- We can define the conditioning operation as:
-- select norm(bp) | norm(0.1, 0.2) from patients;

/*-- Converts a bernoulli type to JSON
create function build_bern_json(b bernoulli) returns json as $$
    select json_build_object('obs', b.data, 'prop', b.prop, 'n' b.n);
$$ language sql;

create function bern_add_sample(b bernoulli, next boolean, prop real) returns bernoulli as $$
    select array_append(b.data, next) as data, prop;
$$ language sql;

create aggregate bernoulli(next boolean, prop real) (
    sfunc = bern_add_sample,
    stype = bernoulli
);*/

-- Normal
-- select norm(NULL, 0.0, 1.0); will create a hidden variable.

/*create type normal as (
    data real[],
    mean real,
    var real
);

create function build_normal(d normal) returns json as $$
    select json_build_object('obs', d.data, 'mean', d.mean, 'var', d.var);
$$ language sql;

create function agg_normal(d normal, next real, mean real, var real) returns normal as $$
    select array_append(d.data, next) as data, mean, var;
$$ language sql;

create aggregate norm(next real, mean real, var real) (
    sfunc = agg_normal,
    stype = normal,
    finalfunc = build_normal
);*/


-- Aggregates into JSON
-- create aggregate bern(next boolean, prop real) (
--    sfunc = agg_bern,
--    stype = bernoulli,
--    finalfunc = build_bern_json
-- );


-- Example usage:
-- create table obs(p boolean);
-- insert into obs values ('t'), ('f'), ('t'), ('f');

-- Represent prior
-- select bernoulli(t, 0.2) from obs;

-- Represent posterior for given prior.
-- select posterior(bernoulli(t, 0.2)) from obs;
-- Poisson

/*create type poisson as (
    data integer[],
    rate real
);

create function build_poiss(p poisson) returns json as $$
    select json_build_object('obs', p.data, 'rate', p.rate);
$$ language sql;

create function agg_poiss(p poisson, next integer, rate real) returns poisson as $$
    select array_append(p.data, next) as data, rate;
$$ language sql;

create aggregate poiss(next integer, rate real) (
    sfunc = agg_poiss,
    stype = poisson,
    finalfunc = build_poiss
);*/

/*-- Create a single factor with informed correlation
create function factor(distr json, corr real) returns json as $$

$$ language sql;

-- Create a set of factors with informed correlations, or NULL.
create function factors(distr json[], corrs real[]) returns json as $$

$$ language sql;

-- Condition the informed factors into the given parent, expanding the probabilistic graph.
create function cond(factors json, parent json) returns json as $$

$$ language sql;

-- C-level functions that evaluate distributions

create function log_prob(distr json) returns double precision as $$
    select log_prob(cast(distr as text));
$$ language sql;

create function prob(distr text) returns real as $$
    'log_prob', 'pgbayes.so'
$$ language sql;

create function mean(distr json) returns real as $$
    'mean', 'pgbayes.so'
$$ language c strict;

-- Returns credible interval for all distribution elements in the form {low, high}
create function credint(distr json, hdi real) returns json as $$
    'cred_interval', 'pgbayes.so'
$$ language c strict;

create function posterior(distr json) returns json as $$
    'posterior' 'pgbayes.so'
$$ language c strict;

create function predict(distr json, vals json) returns json as $$
    'predict' 'pgbayes.so'
$$ language c strict;

-- Generates a new random sample for this model, using the current parameters.
create function sample(distr json) returns json as $$
    'sample', 'pgbayes.so'
$$ language c strict;

-- Returns AUC and ROC measures for k-fold cross-validation over a discrete outcome.
create function cross_validate(distr json, n integer) returns json as $$
    'cross_validate', 'pgbayes.so'
$$ language c strict;

-- Modifies the informed probabilistic model by setting the packed observation obs
-- as the data of the model, re-mapping all names present in obs at the model. If names are not
-- found, then the variable is set as NULL and is re-sampled.
-- If the model represens a posterior, a way to make
-- new predictions is setting observations to the dependent variables:
-- set_sample(distr, '{x1 : {1.0...N}, x2 : {2.0..N}}');
-- sample(distr).
create function set_sample(distr json, obs json) returns json as $$

$$ language c strict;*/
