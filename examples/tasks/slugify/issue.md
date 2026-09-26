Title: slugify() produces repeated and trailing hyphens

`slugify("  Hello,   World!! ")` returns `"--hello----world---"`.
Slugs are used in URLs, so they should look like `"hello-world"`: lowercase words
joined by a single hyphen, with no hyphens at the start or end.
