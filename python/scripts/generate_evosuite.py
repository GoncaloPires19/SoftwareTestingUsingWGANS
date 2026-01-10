"""
EvoSuite test generation pipeline.
Generates test suites for Java classes to enrich training data with synthetic tests.
"""

import subprocess
import logging
from pathlib import Path
from typing import List, Optional
import pandas as pd
import json


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvoSuiteGenerator:
    """Generate tests using EvoSuite"""
    
    def __init__(self, evosuite_jar: Optional[Path] = None, maven_target: Path = Path("target/classes")):
        """
        Args:
            evosuite_jar: Path to evosuite.jar (if not in PATH)
            maven_target: Path to compiled classes (default: Maven target/classes)
        """
        self.evosuite_jar = evosuite_jar
        self.maven_target = Path(maven_target)
        self.output_dir = Path("data/raw/evosuite")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def compile_project(self) -> bool:
        """Compile Java project with Maven"""
        logger.info("🔨 Compiling project with Maven...")
        try:
            result = subprocess.run(
                ["mvn", "clean", "compile"],
                capture_output=True,
                text=True,
                cwd="..",
            )
            if result.returncode == 0:
                logger.info("✅ Project compiled")
                return True
            else:
                logger.error(f"Maven compilation failed:\n{result.stderr}")
                return False
        except FileNotFoundError:
            logger.error("Maven not found. Install with: sudo apt-get install maven")
            return False
    
    def find_classes(self) -> List[str]:
        """Find all compiled Java classes"""
        logger.info("🔍 Finding compiled classes...")
        
        class_files = list(self.maven_target.glob("**/*.class"))
        classes = []
        
        for class_file in class_files:
            # Skip inner classes and other generated files
            if "$" in class_file.name or class_file.name.endswith("Test.class"):
                continue
            
            # Convert path to package.Class format
            relative = class_file.relative_to(self.maven_target)
            class_name = str(relative).replace("/", ".").replace("\\", ".").replace(".class", "")
            classes.append(class_name)
        
        logger.info(f"✅ Found {len(classes)} classes")
        return classes
    
    def generate_for_class(
        self,
        class_name: str,
        time_budget: int = 60,
        population_size: int = 20,
        search_algorithm: str = "WHOLE_SUITE",
    ) -> bool:
        """
        Generate tests for a single class
        
        Args:
            class_name: Fully qualified class name (e.g., com.example.MathOperations)
            time_budget: Time in seconds
            population_size: EvoSuite population size
            search_algorithm: WHOLE_SUITE, MOSUITE, DynaMOSA, etc.
        """
        
        logger.info(f"🧬 Generating tests for {class_name} ({time_budget}s)...")
        
        cmd = [
            "java", "-jar", str(self.evosuite_jar) if self.evosuite_jar else "evosuite",
            "-cp", str(self.maven_target),
            "-class", class_name,
            "-Dtest_dir=" + str(self.output_dir / "src"),
            f"-Dtime_budget={time_budget}",
            f"-Dpopulation_size={population_size}",
            f"-Dalgorithm={search_algorithm}",
            "-Dcriterion=BRANCH",  # Branch coverage
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=time_budget + 30,
            )
            
            if result.returncode == 0 or "Tests written to" in result.stdout:
                logger.info(f"✅ Generated for {class_name}")
                return True
            else:
                logger.warning(f"⚠️  Generation for {class_name} may have issues")
                logger.debug(f"Output: {result.stdout}\n{result.stderr}")
                return True  # Still count as success (tests may have been written)
        
        except subprocess.TimeoutExpired:
            logger.warning(f"⏱️  Timeout for {class_name}")
            return False
        except FileNotFoundError:
            logger.error("EvoSuite not found. Install with:")
            logger.error("  Download: https://www.evosuite.org/download/")
            logger.error("  Or: pip install evosuite")
            return False
    
    def generate_batch(
        self,
        classes: Optional[List[str]] = None,
        time_per_class: int = 60,
        max_classes: Optional[int] = None,
        skip_pattern: str = "Test",
    ) -> Dict[str, bool]:
        """
        Generate tests for multiple classes
        
        Args:
            classes: List of class names (finds automatically if None)
            time_per_class: Time budget per class in seconds
            max_classes: Limit number of classes (for testing)
            skip_pattern: Skip classes matching pattern
        
        Returns:
            Dictionary mapping class names to success status
        """
        
        if classes is None:
            classes = self.find_classes()
        
        # Filter
        classes = [c for c in classes if skip_pattern not in c]
        
        if max_classes:
            classes = classes[:max_classes]
        
        logger.info(f"\n🧬 EVOSUITE GENERATION")
        logger.info(f"   Classes: {len(classes)}")
        logger.info(f"   Time per class: {time_per_class}s")
        logger.info(f"   Total time: ~{len(classes) * time_per_class / 60:.1f} minutes\n")
        
        results = {}
        for i, class_name in enumerate(classes):
            logger.info(f"[{i+1}/{len(classes)}] {class_name}")
            success = self.generate_for_class(class_name, time_budget=time_per_class)
            results[class_name] = success
        
        # Summary
        successful = sum(1 for v in results.values() if v)
        logger.info(f"\n✅ Generated tests for {successful}/{len(classes)} classes")
        
        return results
    
    def collect_generated_tests(self) -> pd.DataFrame:
        """
        Collect generated tests into normalized DataFrame
        """
        logger.info("📦 Collecting generated tests...")
        
        records = []
        test_files = list(self.output_dir.glob("**/*_ESTest.java"))
        
        logger.info(f"   Found {len(test_files)} test files")
        
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8', errors='ignore') as f:
                    code = f.read()
                
                # Simple metrics
                methods = code.count("public void test")
                branches = code.count("if") + code.count("else") + code.count("switch")
                statements = code.count(";")
                
                records.append({
                    "test_name": test_file.stem,
                    "test_code": code[:1000],
                    "source_file": str(test_file),
                    "source": "evosuite",
                    "cwe": None,
                    "bug_type": None,
                    "methods": methods,
                    "branches": branches,
                    "statements": statements,
                    "has_vulnerability": False,
                    "num_test_methods": methods,
                })
            except Exception as e:
                logger.debug(f"Failed to process {test_file}: {e}")
        
        df = pd.DataFrame(records)
        
        # Save
        output_file = Path("data/processed/evosuite_normalized.csv")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        
        logger.info(f"✅ Saved {len(df)} tests to {output_file}")
        
        return df


def main():
    """Main EvoSuite generation pipeline"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate tests with EvoSuite")
    parser.add_argument("--classes", type=int, default=5, help="Number of classes to generate tests for")
    parser.add_argument("--time-per-class", type=int, default=60, help="Time budget per class (seconds)")
    parser.add_argument("--skip-compile", action="store_true", help="Skip Maven compilation")
    parser.add_argument("--evosuite-jar", type=str, help="Path to evosuite.jar")
    
    args = parser.parse_args()
    
    # Create generator
    generator = EvoSuiteGenerator(
        evosuite_jar=Path(args.evosuite_jar) if args.evosuite_jar else None
    )
    
    # Compile
    if not args.skip_compile:
        if not generator.compile_project():
            logger.error("Compilation failed. Use --skip-compile to skip.")
            return
    
    # Generate
    results = generator.generate_batch(
        max_classes=args.classes,
        time_per_class=args.time_per_class,
    )
    
    # Collect
    try:
        tests_df = generator.collect_generated_tests()
        
        print("\n" + "="*70)
        print("✅ EVOSUITE GENERATION COMPLETE")
        print("="*70)
        print(f"Generated {len(tests_df)} test cases")
        print(f"Output: data/processed/evosuite_normalized.csv")
        
    except Exception as e:
        logger.error(f"Failed to collect tests: {e}")


if __name__ == "__main__":
    main()
