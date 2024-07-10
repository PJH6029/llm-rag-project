# Kendra

## Base Document

```
{
    "Attributes": {
        "_category": "base",
        "version": "v2.5",
    }
}
```

## Additional Document

```
{
    "Attributes": {
        "_category": "additional",
        "version": "v2.5-addendum-v0.20",
        "base-doc-id": "<base-doc-s3-id>"
    }
}
```

# Knowledge Base for Bedrock

## Base Document

```
{
    "metadataAttributes": {
        "category": "base",
        "version": "v2.5"
    }
}
```

## Additional Document

```
{
    "metadataAttributes": {
        "category": "additional",
        "version": "v2.5-addendum-v0.20",
        "base-doc-id": "s3://llm-project-demo-bucket/frequently_access_documents/OCP/2.5/datacenter-nvme-ssd-specification-v2-5-pdf.pdf"
    }
}
```