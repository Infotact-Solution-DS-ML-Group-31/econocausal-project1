"""Unit tests for application configuration settings."""

import unittest
from src.api.config import Settings, get_settings


class TestApiConfig(unittest.TestCase):
    def test_default_settings(self):
        cfg = get_settings()
        self.assertEqual(cfg.app_name, "EconoCausal API")
        self.assertEqual(cfg.port, 8000)
        self.assertEqual(cfg.default_campaign_budget, 10000.0)
        self.assertTrue(cfg.debug)

    def test_custom_settings_override(self):
        custom_cfg = Settings(
            app_name="Custom EconoCausal",
            port=9000,
            default_campaign_budget=25000.0,
            environment="production",
            debug=False,
        )
        self.assertEqual(custom_cfg.app_name, "Custom EconoCausal")
        self.assertEqual(custom_cfg.port, 9000)
        self.assertEqual(custom_cfg.default_campaign_budget, 25000.0)
        self.assertEqual(custom_cfg.environment, "production")
        self.assertFalse(custom_cfg.debug)


if __name__ == "__main__":
    unittest.main()