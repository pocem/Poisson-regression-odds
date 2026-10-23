-- Bets are placed at exactly the odds the website shows.
-- Run once in the Supabase dashboard (SQL Editor -> New query -> paste -> Run), after
-- schema.sql and leaderboard.sql. Safe to run again.
--
-- match_odds      The Bet365 odds (and the model's chances) of every upcoming match as shown on
--                 the site, uploaded by the GitHub workflow after each build
--                 (src/site/sync_results.py, with the secret key).
-- lock_bet_odds   Runs on every new bet and overwrites its odds, kickoff and model chance with
--                 the site's values, so a bet can't be logged at any other price -- not even by
--                 calling the API directly. It rejects matches without odds and bets after the
--                 match has started.

create table if not exists public.match_odds (
  league     text not null,
  season     text not null,
  home       text not null,
  away       text not null,
  round      int,
  kickoff    timestamptz not null,
  locks_at   timestamptz not null,  -- kickoff; end of the weekend when the kickoff time is TBC
  odds_h     numeric(7, 2) not null,
  odds_d     numeric(7, 2) not null,
  odds_a     numeric(7, 2) not null,
  model_h    numeric(6, 4),
  model_d    numeric(6, 4),
  model_a    numeric(6, 4),
  updated_at timestamptz not null default now(),
  primary key (league, season, home, away)
);

alter table public.match_odds enable row level security;

drop policy if exists "Signed-in users read match odds" on public.match_odds;
create policy "Signed-in users read match odds" on public.match_odds
  for select to authenticated using (true);

revoke all on public.match_odds from anon;
grant select on public.match_odds to authenticated;
grant all on public.match_odds to service_role;

create or replace function public.lock_bet_odds()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
declare
  o public.match_odds;
begin
  select * into o from public.match_odds
  where league = new.league and season = new.season and home = new.home and away = new.away;
  if not found then
    raise exception 'Bet365 odds for this match aren''t on the site yet, so it can''t be logged.';
  end if;
  if o.locks_at <= now() then
    raise exception 'This match has already kicked off.';
  end if;
  new.odds         := case new.outcome when 'H' then o.odds_h when 'D' then o.odds_d else o.odds_a end;
  new.bet365_odds  := new.odds;
  new.model_chance := case new.outcome when 'H' then o.model_h when 'D' then o.model_d else o.model_a end;
  new.kickoff      := o.kickoff;
  new.round        := o.round;
  new.status       := 'bet';
  new.created_at   := now();
  return new;
end;
$$;

drop trigger if exists lock_bet_odds on public.bets;
create trigger lock_bet_odds
  before insert on public.bets
  for each row execute function public.lock_bet_odds();

-- A bet can be deleted only before its match kicks off (the kickoff comes from match_odds,
-- set by the trigger above), so a losing bet can't be removed from the record afterwards.
drop policy if exists "Users delete their own bets" on public.bets;
create policy "Users delete their own bets" on public.bets
  for delete to authenticated using ((select auth.uid()) = user_id and kickoff > now());
