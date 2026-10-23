-- Leaderboard: scores every logged bet against the real results.
-- Run once in the Supabase dashboard (SQL Editor -> New query -> paste -> Run), after schema.sql.
-- Safe to run again.
--
-- results      Match results, uploaded by the GitHub workflow after every site build
--              (src/site/sync_results.py, with the project's secret key, which bypasses
--              row level security). Signed-in users can read them; nobody else can write.
-- leaderboard  Per-player totals of bets placed (skipped picks don't count). Runs with the
--              owner's rights so it can see everyone's bets, but returns totals only -- never
--              individual bets -- and only to signed-in users.
--
-- Score = profit at a flat 1-unit stake: a win at odds 3.00 scores +2, a loss -1. Stakes are
-- ignored, so a big stake can't buy the top spot.

create table if not exists public.results (
  league     text not null,
  season     text not null,
  home       text not null,
  away       text not null,
  kickoff    timestamptz,
  result     text not null check (result in ('H', 'D', 'A')),
  home_goals int,
  away_goals int,
  updated_at timestamptz not null default now(),
  primary key (league, season, home, away)
);

alter table public.results enable row level security;

drop policy if exists "Signed-in users read results" on public.results;
create policy "Signed-in users read results" on public.results
  for select to authenticated using (true);

revoke all on public.results from anon;
grant select on public.results to authenticated;
grant all on public.results to service_role;

create or replace function public.leaderboard()
returns table (
  player        text,
  is_me         boolean,
  bets          int,
  settled       int,
  won           int,
  score         numeric,
  avg_odds      numeric,
  first_kickoff timestamptz
)
language sql
stable
security definer
set search_path = ''
as $$
  select
    coalesce(nullif(trim(u.raw_user_meta_data ->> 'display_name'), ''), 'Unnamed player') as player,
    b.user_id = auth.uid()                                              as is_me,
    count(*)::int                                                       as bets,
    count(r.result)::int                                                as settled,
    count(*) filter (where r.result = b.outcome)::int                   as won,
    coalesce(sum(case when r.result is null then 0
                      when r.result = b.outcome then b.odds - 1
                      else -1 end), 0)                                  as score,
    round(avg(b.odds), 2)                                               as avg_odds,
    min(b.kickoff)                                                      as first_kickoff
  from public.bets b
  join auth.users u on u.id = b.user_id
  left join public.results r
    on r.league = b.league and r.season = b.season and r.home = b.home and r.away = b.away
  where b.status = 'bet'
  group by b.user_id, u.raw_user_meta_data
  order by score desc, settled desc;
$$;

revoke all on function public.leaderboard() from public, anon;
grant execute on function public.leaderboard() to authenticated;
