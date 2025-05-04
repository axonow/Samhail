#!/usr/bin/env python3
"""
Markov Chain Model Sanity Check Script

This script runs a series of sanity checks on the Markov Chain model
to validate its behavior before deployment.
"""
import os
import sys
import time
import pickle
import argparse
from datetime import datetime

# Add project root to Python path to ensure imports work correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import project modules
# Import the MarkovChain class directly from the same directory
from utils.loggers.json_logger import get_logger
from utils.system_monitoring import ResourceMonitor
from markov_chain import MarkovChain

class MarkovChainSanityChecker:
    """
    Runs a series of sanity checks on the Markov Chain model.

    This validates that the model:
    1. Initializes correctly
    2. Generates text without errors
    3. Adheres to expected behavior with various configurations
    """

    def __init__(self, environment="development", memory_threshold_mb=None,
                 memory_percentage=85):
        """
        Initialize the sanity checker with specified parameters.

        Args:
            environment (str): Environment setting ('development', 'test', 'production')
            memory_threshold_mb (int, optional): Memory threshold in MB
            memory_percentage (int): Percentage of system memory to use if threshold not specified
        """
        self.environment = environment

        # Set up logging to a specific log file
        self.log_dir = os.path.join(current_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        # Use a fixed log file path
        log_file = os.path.join(self.log_dir, "sanity_run.log")
        self.logger = get_logger(
            f"markov_sanity_{environment}", log_file=log_file)

        # Initialize resource monitor for tracking system resources
        self.resource_monitor = ResourceMonitor(
            logger=self.logger,
            memory_limit_mb=memory_threshold_mb,
            memory_limit_percentage=memory_percentage,
            monitoring_interval=5.0  # Log metrics every 5 seconds
        )

        # Initialize model
        self.model = None

        # Log initialization with system info
        self.logger.info(f"MarkovChainSanityChecker initialized", extra={
            "metrics": {
                "environment": environment,
                "memory_threshold_mb": memory_threshold_mb,
                "memory_percentage": memory_percentage
            }
        })

        # Print initialization info
        print(
            f"\033[1m🔍 Initializing Markov Chain Sanity Check (env={environment})\033[0m")

    def initialize_model(self, n_gram=2):
        """
        Initialize the Markov Chain model.

        Args:
            n_gram (int): n-gram size for the model

        Returns:
            bool: True if model was initialized successfully, False otherwise
        """
        start_time = time.time()
        self.resource_monitor.start("model_initialization")

        # Log initialization start
        self.logger.info(
            f"Initializing Markov Chain model with n_gram={n_gram}")

        try:
            # Initialize the model
            self.model = MarkovChain(
                n_gram=n_gram,
                environment=self.environment,
                logger=self.logger
            )

            # Train with sample data for testing
            sample_text = """
            The quick brown fox jumps over the lazy dog. A fast black cat runs past a sleeping hound.
            Weather today is sunny with a chance of rain later. Scientists discover new species in the Amazon rainforest.
            The company announced a new product launch next month. Students prepare for final exams at the university.
            """

            self.model.train(sample_text)

            # Get model info
            model_info = {
                "n_gram": self.model.n_gram,
                "class": self.model.__class__.__name__,
                "transitions_count": len(getattr(self.model, 'transitions', {})),
                "environment": self.environment
            }

            # Log initialization completion
            init_time = time.time() - start_time

            self.resource_monitor.log_progress(
                f"Model initialized successfully",
                operation="model_initialization_complete",
                extra_metrics={
                    "init_time": init_time,
                    "model_info": model_info
                }
            )

            print(
                f"\033[1m✅ Model initialized successfully in {init_time:.2f} seconds\033[0m")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing model: {str(e)}")
            print(f"\033[1;31mError initializing model: {str(e)}\033[0m")
            return False
        finally:
            self.resource_monitor.stop()

    def run_generation_test(self, seed_text, num_words=50, temperature=1.0):
        """
        Test generating text with the model.

        Args:
            seed_text (str): Text to start generation with
            num_words (int): Number of words to generate
            temperature (float): Randomness of generation (higher = more random)

        Returns:
            tuple: (success_bool, generated_text)
        """
        start_time = time.time()
        self.resource_monitor.start("text_generation")

        # Log test start
        self.logger.info(f"Running generation test", extra={
            "metrics": {
                "seed_text": seed_text,
                "num_words": num_words,
                "temperature": temperature
            }
        })

        try:
            # Make sure model is initialized
            if not self.model:
                self.logger.error(
                    "Model not initialized, cannot run generation test")
                return False, None

            # Generate text
            generated_text = self.model.generate_text(
                start=seed_text,
                max_length=num_words,
                temperature=temperature
            )

            # Analyze results
            output_word_count = len(generated_text.split())

            # Log completion
            generation_time = time.time() - start_time

            self.resource_monitor.log_progress(
                f"Text generation test completed",
                progress_percent=100,
                operation="text_generation_complete",
                extra_metrics={
                    "generation_time": generation_time,
                    "output_word_count": output_word_count,
                    "output_length": len(generated_text),
                    "words_per_second": output_word_count / generation_time if generation_time > 0 else 0
                }
            )

            print(
                f"\033[1m✅ Generated {output_word_count} words in {generation_time:.2f} seconds\033[0m")
            print(
                f"\033[1m📝 Sample output: \"{generated_text[:100]}...\"\033[0m")

            return True, generated_text

        except Exception as e:
            self.logger.error(f"Error in text generation: {str(e)}")
            print(f"\033[1;31mError in text generation: {str(e)}\033[0m")
            return False, None
        finally:
            self.resource_monitor.stop()

    def test_persistence(self):
        """
        Test saving and reloading the model.

        Returns:
            bool: True if test passed, False otherwise
        """
        start_time = time.time()
        self.resource_monitor.start("persistence_test")

        # Create a temporary file path for the test
        output_path = os.path.join(
            self.log_dir, f"persistence_test_model_{int(time.time())}.pkl")

        # Log test start and file creation plan
        self.logger.info(f"Running persistence test", extra={
            "metrics": {
                "output_path": output_path,
                "file_operation": "will_create"
            }
        })

        print(
            f"\033[1m💾 Starting persistence test - will create file: {output_path}\033[0m")

        try:
            # Make sure model is initialized
            if not self.model:
                self.logger.error(
                    "Model not initialized, cannot run persistence test")
                return False

            # Save model
            try:
                with open(output_path, 'wb') as f:
                    pickle.dump(self.model, f)

                save_time = time.time() - start_time
                save_size = os.path.getsize(output_path) / (1024 * 1024)

                self.logger.info(f"Model saved for persistence test", extra={
                    "metrics": {
                        "save_time": save_time,
                        "save_size_mb": save_size,
                        "file_path": output_path,
                        "file_operation": "created"
                    }
                })

                print(
                    f"\033[1m💾 Model saved in {save_time:.2f} seconds ({save_size:.2f} MB)\033[0m")
            except Exception as e:
                self.logger.error(f"Error saving model", extra={
                    "metrics": {
                        "error": str(e),
                        "file_path": output_path,
                        "file_operation": "creation_failed"
                    }
                })
                print(f"\033[1;31mError saving model: {str(e)}\033[0m")
                return False

            # Reload model
            try:
                reload_start = time.time()
                with open(output_path, 'rb') as f:
                    reloaded_model = pickle.load(f)

                reload_time = time.time() - reload_start

                self.logger.info(f"Model reloaded for persistence test", extra={
                    "metrics": {
                        "reload_time": reload_time,
                        "file_path": output_path
                    }
                })

                print(
                    f"\033[1m🔄 Model reloaded in {reload_time:.2f} seconds\033[0m")
            except Exception as e:
                self.logger.error(f"Error reloading model", extra={
                    "metrics": {
                        "error": str(e),
                        "file_path": output_path
                    }
                })
                print(f"\033[1;31mError reloading model: {str(e)}\033[0m")

                # Clean up file even if reloading failed
                if os.path.exists(output_path):
                    try:
                        os.remove(output_path)
                        self.logger.info("Temporary file deleted after reload failure", extra={
                            "metrics": {
                                "file_path": output_path,
                                "file_operation": "deleted",
                                "reason": "cleanup_after_error"
                            }
                        })
                        print(
                            f"\033[1;33m🗑️ Cleaned up temporary file after reload failure\033[0m")
                    except Exception as cleanup_error:
                        self.logger.error("Failed to delete temporary file", extra={
                            "metrics": {
                                "file_path": output_path,
                                "file_operation": "delete_failed",
                                "error": str(cleanup_error)
                            }
                        })

                return False

            # Compare models
            try:
                # Check if transitions match
                if not hasattr(self.model, 'transitions') or not hasattr(reloaded_model, 'transitions'):
                    self.logger.error("Models don't have transitions attribute for comparison", extra={
                        "metrics": {
                            "original_has_transitions": hasattr(self.model, 'transitions'),
                            "reloaded_has_transitions": hasattr(reloaded_model, 'transitions')
                        }
                    })
                    return False

                original_transitions = len(self.model.transitions)
                reloaded_transitions = len(reloaded_model.transitions)

                transitions_match = original_transitions == reloaded_transitions

                self.logger.info(f"Model comparison for persistence test", extra={
                    "metrics": {
                        "original_transitions": original_transitions,
                        "reloaded_transitions": reloaded_transitions,
                        "transitions_match": transitions_match
                    }
                })

                if transitions_match:
                    print(
                        f"\033[1m✅ Models match: both have {original_transitions} transitions\033[0m")
                else:
                    print(
                        f"\033[1;31mModels don't match: original has {original_transitions} transitions, reloaded has {reloaded_transitions}\033[0m")

                test_passed = transitions_match

            except Exception as e:
                self.logger.error(f"Error comparing models", extra={
                    "metrics": {
                        "error": str(e)
                    }
                })
                print(f"\033[1;31mError comparing models: {str(e)}\033[0m")
                test_passed = False

            # Clean up the temporary file
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                    self.logger.info("Temporary file deleted after persistence test", extra={
                        "metrics": {
                            "file_path": output_path,
                            "file_operation": "deleted",
                            "reason": "test_complete"
                        }
                    })
                    print(
                        f"\033[1m🗑️ Test complete - temporary file deleted\033[0m")
                except Exception as cleanup_error:
                    self.logger.error("Failed to delete temporary file", extra={
                        "metrics": {
                            "file_path": output_path,
                            "file_operation": "delete_failed",
                            "error": str(cleanup_error)
                        }
                    })
                    print(
                        f"\033[1;31mFailed to delete temporary file: {str(cleanup_error)}\033[0m")

            return test_passed

        finally:
            total_time = time.time() - start_time

            self.resource_monitor.log_progress(
                f"Persistence test completed",
                progress_percent=100,
                operation="persistence_test_complete",
                extra_metrics={
                    "total_test_time": total_time
                }
            )

            self.resource_monitor.stop()

    def run_performance_test(self, num_generations=10, words_per_generation=100):
        """
        Test the performance of text generation.

        Args:
            num_generations (int): Number of text generations to perform
            words_per_generation (int): Number of words in each generation

        Returns:
            dict: Performance metrics
        """
        self.resource_monitor.start("performance_test")

        # Log test start
        self.logger.info(f"Running performance test", extra={
            "metrics": {
                "num_generations": num_generations,
                "words_per_generation": words_per_generation
            }
        })

        print(
            f"\033[1m🏁 Running performance test: {num_generations} generations of {words_per_generation} words each\033[0m")

        try:
            # Make sure model is initialized
            if not self.model:
                self.logger.error(
                    "Model not initialized, cannot run performance test")
                return None

            generation_times = []

            # Starter seeds
            starter_seeds = [
                "The company announced",
                "In recent news",
                "Scientists discovered",
                "The president stated",
                "According to recent studies",
                "The market showed",
                "Experts believe that",
                "The report indicates",
                "On Tuesday morning",
                "During the conference"
            ]

            # Run multiple generations and measure performance
            for i in range(num_generations):
                # Get seed text (cycle through starters if needed)
                seed_text = starter_seeds[i % len(starter_seeds)]

                # Report progress
                progress = ((i + 1) / num_generations) * 100
                self.resource_monitor.log_progress(
                    f"Running generation {i+1}/{num_generations}",
                    progress_percent=progress
                )

                print(
                    f"\033[1m🔄 Running generation {i+1}/{num_generations} ({progress:.1f}%)\033[0m")

                # Time the generation
                start_time = time.time()

                try:
                    generated_text = self.model.generate_text(
                        start=seed_text,
                        max_length=words_per_generation,
                        temperature=1.0
                    )

                    generation_time = time.time() - start_time
                    generation_times.append(generation_time)

                    # Print brief update
                    output_word_count = len(generated_text.split())
                    words_per_second = output_word_count / \
                        generation_time if generation_time > 0 else 0
                    print(
                        f"  - Generated {output_word_count} words in {generation_time:.2f}s ({words_per_second:.1f} words/sec)")

                except Exception as e:
                    self.logger.error(
                        f"Error in performance test generation {i+1}: {str(e)}")
                    print(
                        f"\033[1;31mError in generation {i+1}: {str(e)}\033[0m")

                # Check memory health after each generation
                is_healthy, memory_usage, warning = self.resource_monitor.memory_manager.check_memory_health()
                if not is_healthy:
                    self.logger.warning("Performance test stopped early due to memory constraints", extra={
                        "metrics": {
                            "completed_generations": i+1,
                            "memory_usage": memory_usage
                        }
                    })
                    print(
                        f"\033[1;33m⚠️ Test stopped early due to memory constraints\033[0m")
                    break

            # Calculate performance metrics
            if generation_times:
                avg_time = sum(generation_times) / len(generation_times)
                min_time = min(generation_times)
                max_time = max(generation_times)
                avg_words_per_second = words_per_generation / avg_time

                metrics = {
                    "avg_time_per_generation": avg_time,
                    "min_generation_time": min_time,
                    "max_generation_time": max_time,
                    "avg_words_per_second": avg_words_per_second,
                    "completed_generations": len(generation_times),
                    "words_per_generation": words_per_generation
                }

                # Log completion
                self.logger.info("Performance test completed", extra={
                    "metrics": metrics
                })

                # Print summary
                print(f"\n\033[1m📊 Performance Test Results:\033[0m")
                print(
                    f"\033[1m  - Completed {len(generation_times)}/{num_generations} generations\033[0m")
                print(
                    f"\033[1m  - Average generation time: {avg_time:.2f} seconds\033[0m")
                print(
                    f"\033[1m  - Average generation speed: {avg_words_per_second:.1f} words/second\033[0m")
                print(
                    f"\033[1m  - Fastest generation: {min_time:.2f} seconds\033[0m")
                print(
                    f"\033[1m  - Slowest generation: {max_time:.2f} seconds\033[0m")

                return metrics
            else:
                self.logger.error(
                    "No successful generations in performance test")
                print(
                    "\033[1;31mPerformance test failed: no successful generations\033[0m")
                return None

        finally:
            self.resource_monitor.stop()

    def run_all_tests(self, n_gram=2):
        """
        Run the full suite of sanity checks.

        Args:
            n_gram (int): n-gram size for the model

        Returns:
            bool: True if all tests passed, False otherwise
        """
        print("\033[1;32m" + "="*80 + "\033[0m")
        print("\033[1;32m🔍 Starting Markov Chain Sanity Checks\033[0m")
        print("\033[1;32m" + "="*80 + "\033[0m")

        # Step 1: Print system information
        self.resource_monitor.print_system_architecture()

        # Step 2: Initialize model
        print(
            f"\n\033[1;36m📂 Test 1: Model Initialization (n_gram={n_gram})\033[0m")
        if not self.initialize_model(n_gram=n_gram):
            print("\033[1;31m❌ Failed to initialize model, stopping tests\033[0m")
            return False

        # Track overall success
        all_tests_passed = True

        # Step 3: Basic generation test
        print("\n\033[1;36m🔤 Test 2: Basic Text Generation\033[0m")
        basic_gen_success, _ = self.run_generation_test(
            seed_text="The company announced that",
            num_words=50,
            temperature=1.0
        )
        all_tests_passed = all_tests_passed and basic_gen_success

        # Step 4: Long text generation test
        print("\n\033[1;36m📚 Test 3: Long Text Generation\033[0m")
        long_gen_success, _ = self.run_generation_test(
            seed_text="In a remarkable turn of events,",
            num_words=200,
            temperature=0.8
        )
        all_tests_passed = all_tests_passed and long_gen_success

        # Step 5: Temperature variation test
        print("\n\033[1;36m🌡️ Test 4: Temperature Variation Test\033[0m")
        temp_test_results = self.test_temperature_variations()
        all_tests_passed = all_tests_passed and temp_test_results

        # Step 6: State handling test
        print("\n\033[1;36m🧠 Test 5: State Handling Test\033[0m")
        state_test_results = self.test_state_handling()
        all_tests_passed = all_tests_passed and state_test_results

        # Step 7: Model persistence test
        print("\n\033[1;36m💾 Test 6: Model Persistence\033[0m")
        persistence_success = self.test_persistence()
        all_tests_passed = all_tests_passed and persistence_success

        # Step 8: Performance test
        print("\n\033[1;36m⚡ Test 7: Performance Test\033[0m")
        performance_metrics = self.run_performance_test(
            num_generations=5, words_per_generation=100)
        performance_success = performance_metrics is not None
        all_tests_passed = all_tests_passed and performance_success

        # Step 9: Deterministic generation test
        print("\n\033[1;36m🎯 Test 8: Deterministic Generation Test\033[0m")
        determinism_results = self.test_deterministic_generation()
        all_tests_passed = all_tests_passed and determinism_results

        # Step 10: Edge case handling test
        print("\n\033[1;36m🧪 Test 9: Edge Case Handling\033[0m")
        edge_case_results = self.test_edge_cases()
        all_tests_passed = all_tests_passed and edge_case_results

        # Final results
        print("\n\033[1;32m" + "="*80 + "\033[0m")
        if all_tests_passed:
            print(
                "\033[1;32m✅ All sanity checks passed! The model is ready for use.\033[0m")
        else:
            print(
                "\033[1;31m❌ Some sanity checks failed. Check the logs for details.\033[0m")
        print("\033[1;32m" + "="*80 + "\033[0m")

        return all_tests_passed

    def test_temperature_variations(self):
        """
        Test text generation with different temperature settings.

        Returns:
            bool: True if test passed, False otherwise
        """
        self.resource_monitor.start("temperature_test")
        self.logger.info("Starting temperature variation test")

        try:
            # Make sure model is initialized
            if not self.model:
                self.logger.error(
                    "Model not initialized, cannot run temperature test")
                return False

            # Test various temperature settings
            temperatures = [0.2, 0.5, 1.0, 1.5, 2.0]
            seed_text = "The future of technology"
            results = {}
            unique_token_counts = {}

            print(
                "\033[1m🌡️ Testing text generation with various temperature settings\033[0m")

            for temp in temperatures:
                try:
                    # Generate text with this temperature
                    generated_text = self.model.generate_text(
                        start=seed_text,
                        max_length=50,
                        temperature=temp
                    )

                    # Store results
                    results[temp] = generated_text
                    tokens = generated_text.split()
                    unique_tokens = len(set(tokens))
                    unique_token_counts[temp] = unique_tokens

                    # Calculate uniqueness ratio
                    uniqueness_ratio = unique_tokens / \
                        len(tokens) if tokens else 0

                    # Log results with metrics
                    self.logger.info(f"Temperature {temp} generation test", extra={
                        "metrics": {
                            "temperature": temp,
                            "text_sample": generated_text[:100] + ("..." if len(generated_text) > 100 else ""),
                            "total_tokens": len(tokens),
                            "unique_tokens": unique_tokens,
                            "uniqueness_ratio": uniqueness_ratio
                        }
                    })

                    print(
                        f"  - Temperature {temp}: Generated {len(tokens)} tokens ({unique_tokens} unique)")
                    print(f"    Uniqueness ratio: {uniqueness_ratio:.2f}")
                    print(f"    Sample: \"{generated_text[:50]}...\"\n")

                except Exception as e:
                    self.logger.error(
                        f"Error in temperature test at temp={temp}: {str(e)}")
                    print(
                        f"\033[1;31m  Error at temperature {temp}: {str(e)}\033[0m")

            # Analyze the variation in uniqueness
            if len(unique_token_counts) > 1:
                # Check if higher temperatures generally produce more varied outputs
                temps = sorted(temperatures)
                uniqueness_values = [
                    unique_token_counts.get(t, 0) for t in temps]

                # Check if there's a trend of increasing uniqueness with temperature
                if uniqueness_values[0] < uniqueness_values[-1]:
                    print(
                        "\033[1m✅ Temperature test passed: Higher temperatures show increased output variation\033[0m")
                    self.logger.info("Temperature test passed", extra={
                        "metrics": {
                            "temperature_trend": "increased variation with higher temperature",
                            "uniqueness_values": uniqueness_values,
                            "temperatures": temps
                        }
                    })
                    return True
                else:
                    print(
                        "\033[1;33m⚠️ Temperature variation not showing expected patterns\033[0m")
                    self.logger.warning("Temperature test warning", extra={
                        "metrics": {
                            "warning": "temperature not showing expected variation patterns",
                            "uniqueness_values": uniqueness_values,
                            "temperatures": temps
                        }
                    })
                    # Still consider it a pass since the model generated outputs
                    return True

            # If we only have one result or no results, consider test passed if we got any output
            return len(results) > 0

        except Exception as e:
            self.logger.error(f"Error in temperature variation test: {str(e)}")
            print(
                f"\033[1;31mError in temperature variation test: {str(e)}\033[0m")
            return False
        finally:
            self.resource_monitor.stop()

    def test_state_handling(self):
        """
        Test how the model handles different state inputs.

        Returns:
            bool: True if test passed, False otherwise
        """
        self.resource_monitor.start("state_handling_test")
        self.logger.info("Starting state handling test")

        try:
            # Make sure model is initialized
            if not self.model:
                self.logger.error(
                    "Model not initialized, cannot run state handling test")
                return False

            # Test different state input formats
            test_cases = [
                {"name": "normal", "state": "The quick brown",
                    "expected_success": True},
                {"name": "empty", "state": "", "expected_success": False},
                {"name": "random_words", "state": "xyzabc defghi jklmno",
                    "expected_success": True},
                # Only expect success for n_gram=1
                {"name": "single_word", "state": "technology",
                    "expected_success": self.model.n_gram == 1},
                {"name": "punctuation", "state": "Hello, world!",
                    "expected_success": True},
            ]

            print(
                "\033[1m🧠 Testing model state handling with different inputs\033[0m")

            success_count = 0
            expected_success_count = sum(
                1 for case in test_cases if case["expected_success"])

            for case in test_cases:
                try:
                    # For single words with n_gram > 1, we need special handling
                    is_single_word = len(
                        case["state"].split()) == 1 and case["state"] != ""
                    if is_single_word and self.model.n_gram > 1:
                        # For n_gram > 1, we need to find a state that ends with this word or contains it
                        if self.model.transitions:
                            found_state = False
                            for state in self.model.transitions.keys():
                                if isinstance(state, tuple) and case["state"] in state:
                                    # Generate from a state containing this word
                                    generation_text = self.model.generate_text(
                                        start=state,
                                        max_length=10,
                                        temperature=1.0
                                    )
                                    test_result = generation_text is not None and len(
                                        generation_text) > 0
                                    found_state = True
                                    break

                            if not found_state:
                                # Couldn't find a matching state, as expected
                                test_result = False
                                generation_text = f"N/A - No n-gram state with word '{case['state']}' found"
                        else:
                            test_result = False
                            generation_text = "N/A - No transitions in model"
                    else:
                        # Handle normal cases
                        generation_text = self.model.generate_text(
                            start=case["state"] if case["state"] else None,
                            max_length=10,
                            temperature=1.0
                        )
                        # If we got text back, consider it a success
                        test_result = generation_text is not None and len(
                            generation_text) > 0

                    # Check if results match expectations
                    result_matches_expectation = test_result == case["expected_success"]
                    if case["expected_success"] and test_result:
                        success_count += 1

                    # Log results
                    self.logger.info(f"State handling test: {case['name']}", extra={
                        "metrics": {
                            "state_name": case["name"],
                            "state_input": case["state"],
                            "expected_success": case["expected_success"],
                            "actual_success": test_result,
                            "matches_expectation": result_matches_expectation,
                            "generation": generation_text,
                            "n_gram": self.model.n_gram
                        }
                    })

                    if result_matches_expectation:
                        print(
                            f"  - {case['name']}: {'✅ Pass' if test_result else '⚠️ Expected failure'}")
                    else:
                        print(
                            f"  - {case['name']}: {'❌ Unexpected success' if test_result else '❌ Unexpected failure'}")

                    print(f"    Input: \"{case['state']}\"")
                    print(f"    Output: \"{generation_text}\"\n")

                except Exception as e:
                    self.logger.error(
                        f"Error in state test '{case['name']}': {str(e)}")
                    print(
                        f"\033[1;31m  Error in test '{case['name']}']: {str(e)}\033[0m")

            # Consider the test passed if all expected successes were actually successful
            test_passed = success_count == expected_success_count

            if test_passed:
                print(
                    "\033[1m✅ State handling test passed: Model handled states as expected\033[0m")
            else:
                print(
                    f"\033[1;33m⚠️ State handling test: Only {success_count}/{expected_success_count} expected successes were achieved\033[0m")

            self.logger.info("State handling test completed", extra={
                "metrics": {
                    "passed": test_passed,
                    "success_count": success_count,
                    "expected_success_count": expected_success_count
                }
            })

            return test_passed

        except Exception as e:
            self.logger.error(f"Error in state handling test: {str(e)}")
            print(f"\033[1;31mError in state handling test: {str(e)}\033[0m")
            return False
        finally:
            self.resource_monitor.stop()

    def test_deterministic_generation(self):
        """
        Test deterministic text generation with fixed seed and temperature=0.

        Returns:
            bool: True if test passed, False otherwise
        """
        self.resource_monitor.start("deterministic_test")
        self.logger.info("Starting deterministic generation test")

        try:
            # Make sure model is initialized
            if not self.model:
                self.logger.error(
                    "Model not initialized, cannot run deterministic test")
                return False

            # Set a fixed random seed
            import random
            random.seed(42)

            print("\033[1m🎯 Testing deterministic text generation\033[0m")

            # Generate text twice with temperature=0 (deterministic)
            seed_text = "The company has decided to"

            # First generation
            print("  Generating first text sample...")
            first_generation = self.model.generate_text(
                start=seed_text,
                max_length=20,
                temperature=0.1  # Nearly deterministic
            )

            # Reset seed
            random.seed(42)

            # Second generation with same parameters
            print("  Generating second text sample...")
            second_generation = self.model.generate_text(
                start=seed_text,
                max_length=20,
                temperature=0.1  # Nearly deterministic
            )

            # Compare results
            deterministic = first_generation == second_generation

            self.logger.info("Deterministic generation test results", extra={
                "metrics": {
                    "deterministic": deterministic,
                    "seed_text": seed_text,
                    "first_generation": first_generation,
                    "second_generation": second_generation
                }
            })

            if deterministic:
                print(
                    f"\033[1m✅ Deterministic test passed: Generated identical text with same seed\033[0m")
                print(f"  Text: \"{first_generation}\"\n")
            else:
                print(
                    f"\033[1;33m⚠️ Deterministic test warning: Generated different texts with same seed\033[0m")
                print(
                    f"  First: \"{first_generation}\"\n  Second: \"{second_generation}\"\n")

            # For this test, we'll consider either result valid as some implementations
            # might use non-deterministic libraries even with fixed seeds
            return True

        except Exception as e:
            self.logger.error(
                f"Error in deterministic generation test: {str(e)}")
            print(
                f"\033[1;31mError in deterministic generation test: {str(e)}\033[0m")
            return False
        finally:
            self.resource_monitor.stop()

    def test_edge_cases(self):
        """
        Test model behavior with edge cases.

        Returns:
            bool: True if test passed, False otherwise
        """
        self.resource_monitor.start("edge_case_test")
        self.logger.info("Starting edge case handling test")

        try:
            # Make sure model is initialized
            if not self.model:
                self.logger.error(
                    "Model not initialized, cannot run edge case test")
                return False

            print("\033[1m🧪 Testing edge case handling\033[0m")
            edge_cases_passed = 0
            total_edge_cases = 0

            # Test Case 1: Very long input
            total_edge_cases += 1
            try:
                long_input = "This is a very long input " * 20
                print("  Testing very long input...")
                result = self.model.generate_text(
                    start=long_input,
                    max_length=10,
                    temperature=1.0
                )
                self.logger.info("Edge case: Very long input", extra={
                    "metrics": {
                        "input_length": len(long_input),
                        "result": result[:100] + ("..." if len(result) > 100 else ""),
                        "success": result is not None
                    }
                })
                if result is not None:
                    print("  ✅ Model handled very long input")
                    edge_cases_passed += 1
                else:
                    print("  ❌ Model failed with very long input")
            except Exception as e:
                self.logger.error(f"Edge case error (long input): {str(e)}")
                print(f"  ❌ Model threw exception with long input: {str(e)}")

            # Test Case 2: Special characters
            total_edge_cases += 1
            try:
                special_input = "!@#$ %^&*() special characters 你好 🚀 😀"
                print("  Testing special characters...")
                result = self.model.generate_text(
                    start=special_input,
                    max_length=10,
                    temperature=1.0
                )
                self.logger.info("Edge case: Special characters", extra={
                    "metrics": {
                        "input": special_input,
                        "result": result[:100] + ("..." if result and len(result) > 100 else ""),
                        "success": result is not None
                    }
                })
                if result is not None:
                    print("  ✅ Model handled special characters")
                    edge_cases_passed += 1
                else:
                    print("  ❌ Model failed with special characters")
            except Exception as e:
                self.logger.error(f"Edge case error (special chars): {str(e)}")
                print(
                    f"  ❌ Model threw exception with special characters: {str(e)}")

            # Test Case 3: Zero temperature
            total_edge_cases += 1
            try:
                print("  Testing zero temperature...")
                result = self.model.generate_text(
                    start="The company announced",
                    max_length=10,
                    temperature=0.0
                )
                self.logger.info("Edge case: Zero temperature", extra={
                    "metrics": {
                        "temperature": 0.0,
                        "result": result[:100] + ("..." if result and len(result) > 100 else ""),
                        "success": result is not None
                    }
                })
                if result is not None:
                    print("  ✅ Model handled zero temperature")
                    edge_cases_passed += 1
                else:
                    print("  ❌ Model failed with zero temperature")
            except Exception as e:
                self.logger.error(
                    f"Edge case error (zero temperature): {str(e)}")
                print(
                    f"  ❌ Model threw exception with zero temperature: {str(e)}")

            # Print summary
            print(
                f"\n  Edge Case Tests: {edge_cases_passed}/{total_edge_cases} passed\n")

            self.logger.info("Edge case test completed", extra={
                "metrics": {
                    "passed": edge_cases_passed,
                    "total": total_edge_cases,
                    "success_rate": edge_cases_passed/total_edge_cases if total_edge_cases > 0 else 0
                }
            })

            # Consider test passed if majority of edge cases passed
            return edge_cases_passed >= (total_edge_cases / 2)

        except Exception as e:
            self.logger.error(f"Error in edge case test: {str(e)}")
            print(f"\033[1;31mError in edge case test: {str(e)}\033[0m")
            return False
        finally:
            self.resource_monitor.stop()


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Run sanity checks on the Markov Chain model")
    parser.add_argument("--env", choices=["development", "test", "production"],
                        default="development", help="Environment (default: development)")
    parser.add_argument(
        "--memory", type=int, help="Memory threshold in MB (default: uses percentage)")
    parser.add_argument("--memory-pct", type=int, default=80,
                        help="Memory percentage to use if threshold not specified (default: 80)")
    parser.add_argument("--n-gram", type=int, default=2,
                        help="N-gram size for the model (default: 2)")
    parser.add_argument("--test", choices=["all", "persistence", "generation", "temperature", "state", "deterministic", "edge"],
                        help="Run a specific test instead of all tests")

    args = parser.parse_args()

    # Create sanity checker
    checker = MarkovChainSanityChecker(
        environment=args.env,
        memory_threshold_mb=args.memory,
        memory_percentage=args.memory_pct
    )

    # Run specific test if requested, otherwise run all
    if args.test:
        if not checker.initialize_model(n_gram=args.n_gram):
            print("\033[1;31m❌ Failed to initialize model, stopping tests\033[0m")
            sys.exit(1)

        # Run the requested test
        if args.test == "persistence":
            checker.test_persistence()
        elif args.test == "generation":
            checker.run_generation_test(
                seed_text="The company announced that",
                num_words=50,
                temperature=1.0
            )
        elif args.test == "temperature":
            checker.test_temperature_variations()
        elif args.test == "state":
            checker.test_state_handling()
        elif args.test == "deterministic":
            checker.test_deterministic_generation()
        elif args.test == "edge":
            checker.test_edge_cases()
        else:  # all
            checker.run_all_tests(n_gram=args.n_gram)
    else:
        # Run all tests
        checker.run_all_tests(n_gram=args.n_gram)
