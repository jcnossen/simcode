"""Acquisitions with several detection channels on one camera (1 = plain SIMCODE, 2 = biplane / two-colour, 4 = 4Pi):
the channel layout of the acquisition config (layout.py), the registration of the channels to the reference channel
(registration.py: the map and its in-situ estimation per file), and the multi-channel front end (front.py: camera
conversion, registered channel sum for the detector, unwarped per-channel patches at the mapped candidate positions)."""
