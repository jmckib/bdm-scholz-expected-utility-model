import argparse
from collections import defaultdict
import csv


class Actor(object):
    """An actor with bounded rationality.

    The methods on this class such as u_success, u_failure, eu_challenge are
    meant to be calculated from the actor's perspective, which in practice
    means that the actor's risk aversion is always used, including to calculate
    utilities for other actors.

    I don't understand why an actor would assume that other actors share the
    same risk aversion, or how this implies that it is from the given actor's
    point of view, but as far as I can tell this is faithful to BDM's original
    formulation as well as Scholz's replication.
    """
    def __init__(self, name, c, s, x, model, r=1.0):
        self.name = name
        self.c = c  # capabilities, float between 0 and 1
        self.s = s  # salience, float between 0 and 1
        self.x = x  # number representing position on an issue
        self.model = model
        self.r = r  # risk aversion, float between .5 and 2

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '%s(x=%s,c=%s,s=%s,r=%.2f)' % (
            self.name, self.x, self.c, self.s, self.r)

    def compare(self, x_j, x_k, risk=None):
        """Difference in utility to `self` between positions x_j and x_k."""
        risk = risk or self.r

        position_range = self.model.position_range
        x_k_distance = (abs(self.x - x_k) / position_range) ** risk
        x_j_distance = (abs(self.x - x_j) / position_range) ** risk
        return self.c * self.s * (x_k_distance - x_j_distance)

    def u_success(self, actor, x_j):
        """Utility to `actor` successfully challenging position x_j."""
        position_range = self.model.position_range
        val = 0.5 - 0.5 * abs(actor.x - x_j) / position_range
        return 2 - 4 * val ** self.r

    def u_failure(self, actor, x_j):
        """Utility to `actor` of failing in challenge position x_j."""
        position_range = self.model.position_range
        val = 0.5 + 0.5 * abs(actor.x - x_j) / position_range
        return 2 - 4 * val ** self.r

    def u_status_quo(self):
        """Utility to `self` of the status quo."""
        return 2 - 4 * (0.5 ** self.r)

    def eu_challenge(self, actor_i, actor_j):
        """Expected utility to `actor_i' of `actor_i` challenging `actor_j`.

        This is calculated from the perspective of actor `self`, which in
        practice means that `self.r` is used for risk aversion.
        """
        prob_success = self.model.probability(actor_i.x, actor_j.x)
        u_success = self.u_success(actor_i, actor_j.x)
        u_failure = self.u_failure(actor_i, actor_j.x)
        u_status_quo = self.u_status_quo()

        eu_resist = actor_j.s * (
            prob_success * u_success + (1 - prob_success) * u_failure)
        eu_not_resist = (1 - actor_j.s) * u_success
        eu_status_quo = self.model.q * u_status_quo

        return eu_resist + eu_not_resist - eu_status_quo

    def danger_level(self):
        """The amount of danger the actor is in from holding its policy position.

        The smaller this number is, the more secure the actor is, in that it
        expects fewer challenges to its position from other actors.
        """
        return sum(self.eu_challenge(other_actor, self) for other_actor
                   in self.model.actors if other_actor != self)

    def risk_acceptance(self):
        """Actor's risk acceptance, based on its current policy position.

        I have two comments:
        - It seems to me that BDM's intent was that in order to calculate
          risk acceptance, one would need to compare an actor's danger level
          across different policy positions that the actor could hold. Instead,
          Scholz compares the actor's danger level to the danger level of all
          other actors. This comparison doesn't seem relevant, given that other
          actors will have danger levels not possible for the given actor
          because of differences in salience and capability.
        - Even (what I assume to be) BDM's original intention is an odd way to
          calculate risk acceptance, given that the actor's policy position may
          have been coerced, rather than having been chosen by the actor based
          on its security preferences.
        """

        # Alternative calculation, which I think is more faithful to
        # BDM's original intent.

        # orig_position = self.x
        # possible_dangers = []
        # for position in self.model.positions():
        #     self.x = position
        #     possible_dangers.append(self.danger_level())
        # self.x = orig_position

        # max_danger = max(possible_dangers)
        # min_danger = min(possible_dangers)

        # return ((2 * self.danger_level() - max_danger - min_danger) /
        #         (max_danger - min_danger))

        danger_levels = [actor.danger_level() for actor in self.model.actors]
        max_danger = max(danger_levels)
        min_danger = min(danger_levels)
        return ((2 * self.danger_level() - max_danger - min_danger) /
                (max_danger - min_danger))

    def risk_aversion(self):
        risk = self.risk_acceptance()
        return (1 - risk / 3.0) / (1 + risk / 3.0)

    def best_offer(self):
        offers = defaultdict(list)

        for other_actor in self.model.actors:
            if self.x == other_actor.x:
                continue

            offer = Offer.from_actors(self, other_actor)
            if offer:
                offers[offer.offer_type].append(offer)

        best_offer = None
        best_offer_key = lambda offer: abs(self.x - offer.position)

        # This is faithful to Scholz' original code, but it appears to be a
        # mistake, since Scholz' paper and BDM clearly state that each actor
        # chooses the offer that requires him to change position the
        # least. Instead, Scholz included a special case for compromises which
        # results in some bizarre behavior, particularly in Round 4 when
        # Belgium compromises with Netherlands to an extreme position rather
        # than with France.
        def compromise_best_offer_key(offer):
            top = (abs(offer.eu) * offer.actor.x +
                   abs(offer.other_eu) * offer.other_actor.x)
            return top / (abs(offer.eu) + abs(offer.other_eu))

        if offers['confrontation']:
            best_offer = min(offers['confrontation'], key=best_offer_key)
        elif offers['compromise']:
            best_offer = min(offers['compromise'],
                             key=compromise_best_offer_key)
        elif offers['capitulation']:
            best_offer = min(offers['capitulation'], key=best_offer_key)

        return best_offer


class Offer(object):
    CONFRONTATION = 'confrontation'
    COMPROMISE = 'compromise'
    CAPITULATION = 'capitulation'
    OFFER_TYPES = (
        CONFRONTATION,
        COMPROMISE,
        CAPITULATION,
    )

    def __init__(self, actor, other_actor, offer_type, eu, other_eu, position):
        if offer_type not in self.OFFER_TYPES:
            raise ValueError('offer_type "%s" not in %s'
                             % (offer_type, self.OFFER_TYPES))

        self.actor = actor  # actor receiving the offer
        self.other_actor = other_actor  # actor proposing the offer
        self.offer_type = offer_type
        self.eu = eu
        self.other_eu = other_eu
        self.position = position

    @classmethod
    def from_actors(cls, actor, other_actor):
        eu_ij = actor.eu_challenge(actor, other_actor)
        eu_ji = actor.eu_challenge(other_actor, actor)

        if eu_ji > eu_ij > 0:
            offer_type = cls.CONFRONTATION
            position = other_actor.x
        elif eu_ji > 0 > eu_ij and eu_ji > abs(eu_ij):
            offer_type = cls.COMPROMISE
            concession = (other_actor.x - actor.x) * abs(eu_ij / eu_ji)
            position = actor.x + concession
        elif eu_ji > 0 > eu_ij and eu_ji < abs(eu_ji):
            offer_type = cls.CAPITULATION
            position = other_actor.x
        else:
            return None

        return cls(actor, other_actor, offer_type, eu_ij, eu_ji, position)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        type_to_fmt = {
            self.CONFRONTATION: '%s loses confrontation to %s',
            self.COMPROMISE: '%s compromises with %s',
            self.CAPITULATION: '%s capitulates to %s',
        }
        fmt = type_to_fmt[self.offer_type] + "\n\t%s vs %s\n\tnew_pos = %s"

        return fmt % (self.actor.name, self.other_actor.name, self.eu,
                      self.other_eu, self.position)


class BDMScholzModel(object):
    """An expected utility model for political forecasting."""

    def __init__(self, data, q=1.0):
        self.actors = [
            Actor(name=item['Actor'],
                  c=float(item['Capability']),
                  s=float(item['Salience']),
                  x=float(item['Position']),
                  model=self)
            for item in data]
        self.name_to_actor = {actor.name: actor for actor in self.actors}
        self.q = q
        positions = self.positions()
        self.position_range = max(positions) - min(positions)

    @classmethod
    def from_csv_path(cls, csv_path):
        return cls(csv.DictReader(open(csv_path, 'rU')))

    def actor_by_name(self, name):
        return self.name_to_actor.get(name)

    def __getitem__(self, key):
        return self.name_to_actor.get(key)

    def positions(self):
        return list({actor.x for actor in self.actors})

    def median_position(self):
        positions = self.positions()
        median = positions[0]
        for position in positions[1:]:
            votes = sum(actor.compare(position, median, risk=1.0)
                        for actor in self.actors)
            if votes > 0:
                median = position
        return median

    def mean_position(self):
        return (sum(actor.c * actor.s * actor.x for actor in self.actors) /
                sum(actor.c * actor.s for actor in self.actors))

    def probability(self, x_i, x_j):
        if x_i == x_j:
            return 0.0

        # `sum_all_votes` below is faithful to Scholz' code, but I think it is
        # quite contrary to BDM's intent. Instead, we should have.
        # denominator = sum(actor.compare(x_i, x_j) for actor in self.actors)

        # This would make sure that prob(x_i, x_j) + prob(x_j, x_i) == 1.
        # However, because of the odd way that salience values are used as
        # the probability that an actor will resist a proposal, this results in
        # the actors almost always confronting each other.

        # My theory is that Scholz got around the confrontation problem by
        # introducing this large denominator, causing extremely small
        # probability values. This prevents actors from confronting each other
        # constantly, but the result is comical, in that the challenging actor
        # always has a vanishingly small chance of winning a conflict, yet the
        # challenged actor often gives up without a fight because of low
        # salience.
        sum_all_votes = sum(abs(actor.compare(a1.x, a2.x))
                            for actor in self.actors
                            for a1 in self.actors
                            for a2 in self.actors)
        return (sum(max(0, actor.compare(x_i, x_j)) for actor in self.actors) /
                sum_all_votes)

    def update_risk_aversions(self):
        for actor in self.actors:
            actor.r = 1.0

        actor_to_risk_aversion = [(actor, actor.risk_aversion())
                                  for actor in self.actors]
        for actor, risk_aversion in actor_to_risk_aversion:
            actor.r = risk_aversion

    def update_positions(self):
        actor_to_best_offer = [(actor, actor.best_offer())
                               for actor in self.actors]
        for actor, best_offer in actor_to_best_offer:
            if best_offer:
                print best_offer
                actor.x = best_offer.position

    def run_model(self, num_rounds=1):
        print 'Median position: %s' % self.median_position()
        print 'Mean position: %s' % self.mean_position()

        for round_ in range(1, num_rounds + 1):
            print ''
            print 'ROUND %d' % round_
            self.update_risk_aversions()
            self.update_positions()

            print ''
            print 'Median position: %s' % self.median_position()
            print 'Mean position: %s' % self.mean_position()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'csv_path',
        help='path to csv with input data')
    parser.add_argument(
        'num_rounds',
        help='number of rounds of simulation to run',
        type=int)
    args = parser.parse_args()

    model = BDMScholzModel.from_csv_path(args.csv_path)
    model.run_model(num_rounds=args.num_rounds)
