-- Bet log for the website's accounts (Supabase project rstembhxuncyttzdmssl).
-- Run once in the Supabase dashboard: SQL Editor -> New query -> paste -> Run.
--
-- Accounts themselves live in Supabase Auth (auth.users; passwords stored as bcrypt
-- hashes). This table holds each user's logged picks -- bets placed and picks skipped.
-- Results aren't stored: the website settles each pick from its own match data.
--
-- Row level security: a signed-in user can only read, add and delete their own rows.
-- There is no update policy, so a logged pick can't be edited after the fact
-- (delete and log it again instead). Visitors who aren't signed in can't read anything.

create table if not exists public.bets (
  id           bigint generated always as identity primary key,
  user_id      uuid not null default auth.uid() references auth.users (id) on delete cascade,
  created_at   timestamptz not null default now(),
  league       text not null check (char_length(league) <= 40),
  season       text not null check (char_length(season) <= 10),
  round        int,
  kickoff      timestamptz not null,
  home         text not null check (char_length(home) <= 60),
  away         text not null check (char_length(away) <= 60),
  outcome      text not null check (outcome in ('H', 'D', 'A')),
  status       text not null check (status in ('bet', 'skip')),
  odds         numeric(7, 2) not null check (odds > 1 and odds < 1000),
  stake        numeric(10, 2) not null default 1 check (stake > 0 and stake < 1000000),
  model_chance numeric(6, 4) check (model_chance between 0 and 1),
  bet365_odds  numeric(7, 2),
  note         text check (char_length(note) <= 300),
  unique (user_id, league, season, home, away, outcome)
);

create index if not exists bets_user_kickoff on public.bets (user_id, kickoff desc);

alter table public.bets enable row level security;

drop policy if exists "Users read their own bets" on public.bets;
create policy "Users read their own bets" on public.bets
  for select to authenticated using ((select auth.uid()) = user_id);

drop policy if exists "Users log their own bets" on public.bets;
create policy "Users log their own bets" on public.bets
  for insert to authenticated with check ((select auth.uid()) = user_id);

drop policy if exists "Users delete their own bets" on public.bets;
create policy "Users delete their own bets" on public.bets
  for delete to authenticated using ((select auth.uid()) = user_id);

revoke all on public.bets from anon;
grant select, insert, delete on public.bets to authenticated;
