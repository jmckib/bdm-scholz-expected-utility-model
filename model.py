from collections import namedtuple
import csv
import unittest


class Actor(object):

    def __init__(self, name, c, s, x, r=1):
        self.name = name
        self.c = c  # capabilities, float between 0 and 1
        self.s = s  # salience, float between 0 and 1
        self.x = x  # number representing position on an issue
        self.r = r  # risk aversion, float between .5 and 2

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '%s(x=%s,c=%s,s=%s,r=%.2f)' % (self.name, self.x, self.c, self.s, self.r)


class Model(object):

    def __init__(self, actors, q=1.0):
        self.actors = actors
        self.name_to_actor = {actor.name: actor for actor in actors}
        self.q = q
        positions = [actor.x for actor in actors]
        self.position_range = max(positions) - min(positions)

    @classmethod
    def from_data(cls, data):
        actors = [Actor(name=item['Actor'],
                        c=float(item['Capability']),
                        s=float(item['Salience']),
                        x=float(item['Position']))
                  for item in data]
        return cls(actors)

    @classmethod
    def from_csv_path(cls, csv_path):
        return cls.from_data(csv.DictReader(open(csv_path, 'rU')))

    def actor_by_name(self, name):
        return self.name_to_actor.get(name)

    def __getitem__(self, key):
        return self.name_to_actor.get(key)

    def positions(self):
        return list({actor.x for actor in self.actors})

    def votes(self, x_j, x_k):
        return sum(self.compare(actor, x_j, x_k) / self.position_range
                   for actor in self.actors)

    def median_position(self):
        positions = self.positions()
        median = positions[0]
        for position in positions[1:]:
            if self.votes(position, median) > 0:
                median = position
        return median

    def mean_position(self):
        top = bottom = 0
        return (sum(actor.c * actor.s * actor.x for actor in self.actors) /
                sum(actor.c * actor.s for actor in self.actors))

    def compare(self, actor, x_j, x_k):
        x_k_distance = (abs(actor.x - x_k) / self.position_range) ** actor.r
        x_j_distance = (abs(actor.x - x_j) / self.position_range) ** actor.r
        return actor.c * actor.s * (x_k_distance - x_j_distance)

    def probability(self, x_i, x_j):
        if x_i == x_j:
            return 0.0
        sum_all_votes = sum(abs(self.compare(actor, a1.x, a2.x))
                            for actor in self.actors
                            for a1 in self.actors
                            for a2 in self.actors)
        return (sum(max(0, self.compare(actor, x_i, x_j)) for actor in self.actors) /
                sum_all_votes)

    def u_success(self, actor, x_j, risk_aversion=None):
        risk_aversion = risk_aversion or actor.r
        return 2 - 4 * (0.5 - 0.5 * abs(actor.x - x_j) / self.position_range) ** risk_aversion

    def u_failure(self, actor, x_j, risk_aversion=None):
        risk_aversion = risk_aversion or actor.r
        return 2 - 4 * (0.5 + 0.5 * abs(actor.x - x_j) / self.position_range) ** risk_aversion

    def u_status_quo(self, actor, risk_aversion=None):
        risk_aversion = risk_aversion or actor.r
        return 2 - 4 * (0.5 ** risk_aversion)

    def eu_challenge(self, actor_i, actor_j, perspective=None):
        perspective = perspective or actor_i
        risk_aversion = perspective.r

        prob = self.probability(actor_i.x, actor_j.x)
        u_s = self.u_success(actor_i, actor_j.x, risk_aversion=risk_aversion)
        u_f = self.u_failure(actor_i, actor_j.x, risk_aversion=risk_aversion)
        u_quo = self.u_status_quo(actor_i, risk_aversion=risk_aversion)

        e_resist = actor_j.s * (prob * u_s + (1 - prob) * u_f)
        e_not_resist = (1 - actor_j.s) * u_s

        return e_resist + e_not_resist - self.q * u_quo

    def get_danger(self, actor):
        return sum(self.eu_challenge(actor_j, actor) for actor_j in self.actors
                   if actor_j != actor)
#                   if actor_j.x != actor.x)

    def actor_name_to_danger(self):
        return {actor.name: self.get_danger(actor) for actor in self.actors}

    def risk_acceptance(self, actor):
        # Right now I'm thinking that this isn't the right way to do. Instead
        # of max danger and min danger of all actors, you should calculate the
        # max danger and min danger for this specific actor of having different
        # policy positions. Otherwise, powerful actors will always appear low
        # risk, because they will be at low risk compared to weaker actors.

        # danger = self.get_danger(actor)

        # orig_position = actor.x
        # possible_dangers = []
        # for position in self.positions():
        #     actor.x = position
        #     possible_dangers.append(self.get_danger(actor))
        # actor.x = orig_position

        # max_danger = max(possible_dangers)
        # min_danger = min(possible_dangers)

        # return ((2 * danger - max_danger - min_danger) /
        #         (max_danger - min_danger))

       actor_to_danger = self.actor_name_to_danger()
       max_danger = max(actor_to_danger.values())
       min_danger = min(actor_to_danger.values())
       return ((2 * actor_to_danger[actor.name] - max_danger - min_danger) /
               (max_danger - min_danger))

    def risk_aversion(self, actor):
        risk = self.risk_acceptance(actor)
        return (1 - risk / 3) / (1 + risk / 3)

    def update_actors_risk_aversion(self):
        for actor in self.actors:
            actor.r = 1.0

        actor_to_risk_aversion = [(actor, self.risk_aversion(actor)) for actor in self.actors]
        for actor, risk_aversion in actor_to_risk_aversion:
            actor.r = risk_aversion

    def update_actors_position(self):
        Offer = namedtuple('Offer', ['actor', 'eu', 'actor_eu', 'offer_position'])

        actor_to_best_offer = {}
        for actor in self.actors:
            offers = {'confrontation': [], 'compromise': [], 'capitulation': [], 'stalemate': []}

            for other_actor in self.actors:
                if actor.x == other_actor.x:
                    continue

                eu_ij = self.eu_challenge(actor, other_actor)
                eu_ji = self.eu_challenge(other_actor, actor, perspective=actor)
                if eu_ji > eu_ij > 0:
                    offers['confrontation'].append(Offer(other_actor, eu_ij, eu_ji, other_actor.x))
                elif eu_ji > 0 > eu_ij and eu_ji > abs(eu_ij):
                    concession = (other_actor.x - actor.x) * abs(eu_ij / eu_ji)
                    offer = actor.x + concession
                    offers['compromise'].append(Offer(other_actor, eu_ij, eu_ji, offer))
                elif eu_ji > 0 > eu_ij and eu_ji < abs(eu_ji):
                    offers['capitulation'].append(Offer(other_actor, eu_ij, eu_ji, other_actor.x))
                else:
                    offers['stalemate'].append(Offer(other_actor, eu_ij, eu_ji, other_actor.x))

            best_offer = None
            if offers['confrontation']:
                # best_offer = max(offers['confrontation'], key=lambda offer: offer.actor_eu)
                best_offer = min(offers['confrontation'], key=lambda offer: abs(actor.x - offer.offer_position))
                print '%s loses confrontation to %s\tnew_pos = %s\t%s vs %s' % (actor, best_offer.actor, best_offer.offer_position, best_offer.eu, best_offer.actor_eu)
            elif offers['compromise']:
                # best_offer = max(offers['compromise'], key=lambda offer: offer.actor_eu)
                best_offer = min(offers['compromise'], key=lambda offer: abs(actor.x - offer.offer_position))
                print '%s compromises with %s\tnew_pos = %s\t%s vs %s' % (actor, best_offer.actor, best_offer.offer_position, best_offer.eu, best_offer.actor_eu)
            elif offers['capitulation']:
                # best_offer = max(offers['capitulation'], key=lambda offer: offer.actor_eu)
                best_offer = min(offers['capitulation'], key=lambda offer: abs(actor.x - offer.offer_position))
                print '%s capitulates to %s\tnew_pos = %s\t%s vs %s' % (actor, best_offer.actor, best_offer.offer_position, best_offer.eu, best_offer.actor_eu)

            if best_offer:
                actor_to_best_offer[actor.name] = best_offer

        for actor in self.actors:
            best_offer = actor_to_best_offer.get(actor.name)
            if best_offer:
                actor.x = actor_to_best_offer[actor.name].offer_position

# model = Model.from_csv_path('ExampleActors.csv')
# print model.median_position(), model.mean_position()
# print ''

# for i in range(1):
#    model.update_actors_risk_aversion()
#    model.update_actors_position()
#    print model.median_position(), model.mean_position()
#    print ''


class TestModel(unittest.TestCase):

    def setUp(self):
        self.data = [
            {'Actor': 'US',
             'Capability': .05,
             'Salience': 10,
             'Position': 0},
            {'Actor': 'China',
             'Capability': .05,
             'Salience': 10,
             'Position': 10},
        ]
        self.model = Model.from_data(self.data)

    def test_compare(self):
        actor = self.model.actor_by_name('US')
        self.assertAlmostEqual(actor.compare(2, 7), .025)
        self.assertAlmostEqual(actor.compare(7, 2), -.025)
        self.assertAlmostEqual(actor.compare(0, 5), .025)

        actor.x = 2
        self.assertAlmostEqual(actor.compare(2, 7), .025)
        self.assertAlmostEqual(actor.compare(7, 2), -.025)

        actor.x = 3
        self.assertAlmostEqual(actor.compare(2, 7), .015)
        self.assertAlmostEqual(actor.compare(7, 2), -.015)

    def test_probability(self):
        a1, a2 = self.model.actors
        self.assertAlmostEqual(self.model.probability(a1.x, a2.x), .5)
        self.assertAlmostEqual(self.model.probability(a2.x, a1.x), .5)

        a1.c *= 2
        self.assertAlmostEqual(self.model.probability(a1.x, a2.x), 2.0 / 3)
        self.assertAlmostEqual(self.model.probability(a2.x, a1.x), 1.0 / 3)

        a1.c /= 2
        a2.s = .5
        self.assertAlmostEqual(self.model.probability(a1.x, a2.x), 1.0 / 6)
        self.assertAlmostEqual(self.model.probability(a2.x, a1.x), 5.0 / 6)

    def test_utility(self):
        actor = self.model.actor_by_name('US')
        other_actor = self.model.actor_by_name('China')

        self.assertAlmostEqual(self.model.u_success(actor, actor.x), 0)
        self.assertAlmostEqual(self.model.u_success(actor, other_actor.x), 2)
        self.assertAlmostEqual(self.model.u_success(actor, (actor.x + other_actor.x) / 2.0), 1)

        self.assertAlmostEqual(self.model.u_failure(actor, actor.x), 0)
        self.assertAlmostEqual(self.model.u_failure(actor, other_actor.x), -2)
        self.assertAlmostEqual(self.model.u_failure(actor, (actor.x + other_actor.x) / 2.0), -1)

        actor.r = 2.0

        self.assertAlmostEqual(self.model.u_success(actor, actor.x), 1)
        self.assertAlmostEqual(self.model.u_success(actor, other_actor.x), 2)
        self.assertAlmostEqual(self.model.u_success(actor, (actor.x + other_actor.x) / 2.0), 1.75)

        self.assertAlmostEqual(self.model.u_failure(actor, actor.x), 1)
        self.assertAlmostEqual(self.model.u_failure(actor, other_actor.x), -2)
        self.assertAlmostEqual(self.model.u_failure(actor, (actor.x + other_actor.x) / 2.0), -0.25)

        self.assertAlmostEqual(self.model.u_status_quo(actor), 1)

        actor.r = .5
        self.assertAlmostEqual(self.model.u_status_quo(actor), -0.8284271247461903)


    def test_eu_challenge(self):
        a1, a2 = self.model.actors
        self.assertAlmostEqual(self.model.eu_challenge(a1, a2), 1.8)
        self.assertAlmostEqual(self.model.eu_challenge(a2, a1), 1.8)

        a1.c *= 2
        self.assertAlmostEqual(self.model.eu_challenge(a1, a2), 1.8666666666666667)
        self.assertAlmostEqual(self.model.eu_challenge(a2, a1), 1.7333333333333334)

        a1.c /= 2
        a2.s = .5

        self.assertAlmostEqual(self.model.eu_challenge(a1, a2), 1.0 / 3)
        self.assertAlmostEqual(self.model.eu_challenge(a2, a1), 1.9333333333333333)

        a2.s = .1
        a1.r = 2.0
        self.assertAlmostEqual(self.model.eu_challenge(a1, a2), 0.8)

        a1.r = .5
        self.assertAlmostEqual(self.model.eu_challenge(a1, a2), 2.6284271247461906)

