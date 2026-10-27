import unittest
from utils.event_logic import BasketEventDetector

class BasketEventTests(unittest.TestCase):
    def collect(self, sequence):
        detector=BasketEventDetector(confirm_frames=2,gap_tolerance_s=.5,cooldown_s=1.5)
        events=[]
        for frame,(t,positive) in enumerate(sequence):
            event=detector.update(t,[(None,.9)] if positive else [],frame)
            if event: events.append(event)
        event=detector.flush()
        if event: events.append(event)
        return events

    def test_five_consecutive_detections_count_once(self):
        events=self.collect([(i*.1,True) for i in range(5)]+[(.9,False),(1.0,False)])
        self.assertEqual(len(events),1)
        self.assertEqual(events[0]["frame"],0)

    def test_short_gap_does_not_split_event(self):
        events=self.collect([(0,True),(.1,True),(.3,False),(.4,True),(.5,False),(1.1,False)])
        self.assertEqual(len(events),1)

    def test_later_distinct_basket_can_be_counted(self):
        events=self.collect([(0,True),(.1,True),(.7,False),(2.1,True),(2.2,True),(2.8,False)])
        self.assertEqual(len(events),2)
        self.assertGreater(events[1]["timestamp_s"],events[0]["timestamp_s"])

if __name__=="__main__": unittest.main()
