import unittest
import humidity
import jax.numpy as jnp

class TestHumidityUnit(unittest.TestCase):

    def test_get_qsat(self):
        temp = jnp.array([[[273] * 96] * 48])
        pressure = jnp.array([[[0.5] * 96] * 48])
        sigma = 4

        # Check that qsat is not null.
        self.assertIsNotNone(humidity.get_qsat(temp, pressure, sigma))
    
    def test_spec_hum_to_rel_hum(self):
        temp = jnp.array([[[273] * 96] * 48])
        pressure = jnp.array([[[0.5] * 96] * 48])
        sigma = 4
        qg = jnp.array([[[2] * 96] * 48])
        rh, qsat = humidity.spec_hum_to_rel_hum(temp, pressure, sigma, qg)

        # Check that rh is not null.
        self.assertIsNotNone(rh)
    
    def test_rel_hum_to_spec_hum(self):
        temp = jnp.array([[[273] * 96] * 48])
        pressure = jnp.array([[[0.5] * 96] * 48])
        sigma = 4
        qg = jnp.array([[[2] * 96] * 48])
        rh, qsat = humidity.spec_hum_to_rel_hum(temp, pressure, sigma, qg)
        qa, qsat = humidity.rel_hum_to_spec_hum(temp, pressure, sigma, rh)

        # Check that qa is the same when converted to rh then back again.
        self.assertEqual(float(jnp.take(qa, 0)), float(jnp.take(qg, 0)))

unittest.main()