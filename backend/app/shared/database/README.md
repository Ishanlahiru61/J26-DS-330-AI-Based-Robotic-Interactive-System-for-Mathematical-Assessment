# Database

This boundary will own the shared database session, migrations, and repository
adapters. Domain modules must depend on repository interfaces, not database
implementation details.
