# EDA Decisions Log

Target Feature: perceived_stress_score

| Feature                        | Kept? | Reason                                             |
| :----------------------------- | :---: | :------------------------------------------------- |
| user_id                        |  No   | Identifier, no predictive signal                   |
| app_name                       |  No   | Single value, no predictive signal                 |
| age                            |  Yes  | Strong correlation with target in profiling report |
| gender                         |  Yes  | Strong correlation with target in profiling report |
| country                        |  Yes  | Strong correlation with target in profiling report |
| urban_rural                    |  Yes  | Strong correlation with target in profiling report |
| income_level                   |  Yes  | Strong correlation with target in profiling report |
| employment_status              |  Yes  | Strong correlation with target in profiling report |
| education_level                |  Yes  | Strong correlation with target in profiling report |
| relationship_status            |  Yes  | Strong correlation with target in profiling report |
| has_children                   |  Yes  | Strong correlation with target in profiling report |
| gender                         |  Yes  | Strong correlation with target in profiling report |
| excercise_hours_per_week       |  Yes  | Strong correlation with target in profiling report |
| sleep_hours_per_night          |  Yes  | Strong correlation with target in profiling report |
| diet_quality                   |  Yes  | Strong correlation with target in profiling report |
| smoking                        |  Yes  | Strong correlation with target in profiling report |
| alcohol_frequency              |  Yes  | Strong correlation with target in profiling report |
| perceived_stress_score         |  Yes  | Strong correlation with target in profiling report |
| self_reported_happiness        |  Yes  | Strong correlation with target in profiling report |
| body_mass_index                |  Yes  | Strong correlation with target in profiling report |
| blood_pressure_systolic        |  Yes  | Strong correlation with target in profiling report |
| blood_pressure_diastolic       |  Yes  | Strong correlation with target in profiling report |
| daily_steps_count              |  Yes  | Strong correlation with target in profiling report |
| weekly_work_hours              |  Yes  | Strong correlation with target in profiling report |
| hobbies_count                  |  Yes  | Strong correlation with target in profiling report |
| social_events_per_month        |  Yes  | Strong correlation with target in profiling report |
| books_read_per_year            |  Yes  | Strong correlation with target in profiling report |
| volunteer_hours_per_month      |  Yes  | Strong correlation with target in profiling report |
| travel_frequency_per_year      |  Yes  | Strong correlation with target in profiling report |
| daily_active_minutes_instagram |  Yes  | Strong correlation with target in profiling report |
| sessions_per_day               |  Yes  | Strong correlation with target in profiling report |
| posts_created_per_week         |  Yes  | Strong correlation with target in profiling report |
| reels_watched_per_day          |  Yes  | Strong correlation with target in profiling report |
| stories_viewed_per_day         |  Yes  | Strong correlation with target in profiling report |
| likes_given_per_day            |  Yes  | Strong correlation with target in profiling report |
| comments_written_per_day       |  Yes  | Strong correlation with target in profiling report |
| dms_sent_per_week              |  Yes  | Strong correlation with target in profiling report |
| dms_received_per_week          |  Yes  | Strong correlation with target in profiling report |
| ads_viewed_per_day             |  Yes  | Strong correlation with target in profiling report |
| ads_clicked_per_day            |  Yes  | Strong correlation with target in profiling report |
| time_on_feed_per_day           |  Yes  | Strong correlation with target in profiling report |
| time_on_explore_per_day        |  Yes  | Strong correlation with target in profiling report |
| time_on_messages_per_day       |  Yes  | Strong correlation with target in profiling report |
| time_on_reel_per_day           |  Yes  | Strong correlation with target in profiling report |
| followers_count                |  Yes  | Strong correlation with target in profiling report |
| following_count                |  Yes  | Strong correlation with target in profiling report |
| uses_premium_features          |  Yes  | Strong correlation with target in profiling report |
| notification_response_rate     |  Yes  | Strong correlation with target in profiling report |
| account_creation_year          |  Yes  | Strong correlation with target in profiling report |
| last_login_date                |  Yes  | Strong correlation with target in profiling report |
| average_session_length_minutes |  Yes  | Strong correlation with target in profiling report |
| content_type_preference        |  Yes  | Strong correlation with target in profiling report |
| preferred_content_theme        |  Yes  | Strong correlation with target in profiling report |
| privacy_setting_level          |  Yes  | Strong correlation with target in profiling report |
| two_factor_auth_enabled        |  Yes  | Strong correlation with target in profiling report |
| biometric_login_used           |  Yes  | Strong correlation with target in profiling report |
| linked_accounts_count          |  Yes  | Strong correlation with target in profiling report |
| subscription_status            |  Yes  | Strong correlation with target in profiling report |
| user_engagement_score          |  Yes  | Strong correlation with target in profiling report |
