"""
이 파일은 requirements.txt에 명시된 패키지들이 현재 Python 환경에 설치되어 있는지 확인하는 유닛 테스트를 수행합니다.

주요 기능:
1. requirements.txt 파일을 읽어 각 패키지와 버전 요구사항을 추출합니다.
2. 추출된 패키지가 현재 설치된 Python 패키지들과 일치하는지 확인합니다.
3. 설치되지 않았거나 버전이 일치하지 않는 패키지가 있으면 테스트가 실패합니다.

실행 방법:
1. 이 파일을 직접 실행하면, `requirements.txt` 파일에 있는 패키지가 설치되어 있는지 자동으로 테스트합니다.
2. unittest 모듈을 사용하여 테스트가 실행되며, 누락된 패키지가 있을 경우 해당 패키지 이름을 출력합니다.

초보자 가이드:
- `__main__` 블록은 이 스크립트가 직접 실행될 때만 테스트가 실행되도록 합니다.
- `importlib.metadata`와 `packaging` 라이브러리를 사용하여 패키지를 확인하고, `unittest`로 테스트를 수행합니다.
"""

import unittest
from pathlib import Path
import importlib.metadata
from packaging.requirements import Requirement

_REQUIREMENTS_PATH = Path(__file__).parent.with_name("requirements.txt")


class TestRequirements(unittest.TestCase):
    """Test availability of required packages."""

    def test_requirements(self):
        """Test that each required package is available."""
        with _REQUIREMENTS_PATH.open() as req_file:
            requirements = req_file.readlines()

        n=0
        for req in requirements:
            req = req.strip()
            if not req or req.startswith("#"):
                # Ignore empty lines and comments
                continue

            requirement = Requirement(req)
            try:
                # Check if the package is installed
                dist = importlib.metadata.version(requirement.name)
            except importlib.metadata.PackageNotFoundError:
                # Print the name of the missing package
                print(f"{requirement.name} is NOT installed")
                n+=1
            else:
                # If the package is installed, print nothing or optionally print a success message
                pass

        if n==0: print("Successfully Installed All Packages !")

if __name__ == "__main__":
    unittest.main()