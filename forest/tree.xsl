<?xml version="1.0"?>
<!-- SPDX-License-Identifier: CC0-1.0 -->
<xsl:stylesheet version="1.0"
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:fr="http://www.jonmsterling.com/jms-005P.xml">

  <xsl:key name="tree-with-uri" match="/fr:tree/fr:mainmatter//fr:tree" use="fr:frontmatter/fr:uri/text()" />

  <xsl:template match="/">
    <html xmlns="http://www.w3.org/1999/xhtml" data-base-url="{/fr:tree/@base-url}">
      <head>
        <meta name="viewport" content="width=device-width" />
        <link rel="stylesheet" href="{/fr:tree/@base-url}style.css" />
        <link rel="stylesheet" href="{/fr:tree/@base-url}katex.min.css" />
        <script type="text/javascript">
          <xsl:if test="/fr:tree/fr:frontmatter/fr:source-path">
            <xsl:text>window.sourcePath = '</xsl:text>
            <xsl:value-of select="/fr:tree/fr:frontmatter/fr:source-path" />
            <xsl:text>'</xsl:text>
          </xsl:if>
        </script>
        <script type="module" src="{/fr:tree/@base-url}forester.js"></script>
        <title>
          <xsl:value-of select="/fr:tree/fr:frontmatter/fr:title/@text" />
        </title>
      </head>
      <body>
        <ninja-keys placeholder="Start typing a note title or ID"></ninja-keys>
        <xsl:if test="not(/fr:tree[@root = 'true'])">
          <header class="header">
            <nav class="nav">
              <div class="logo">
                <a href="{/fr:tree/@base-url}index.html" title="Home">
                  <xsl:text>« Home</xsl:text>
                </a>
              </div>
            </nav>
          </header>
        </xsl:if>
        <div id="grid-wrapper">
          <article>
            <xsl:apply-templates select="fr:tree" />
          </article>
          <xsl:if test="fr:tree/fr:mainmatter/fr:tree[not(@toc='false')] and not(/fr:tree/fr:frontmatter/fr:meta[@name = 'toc']/.='false')">
            <nav id="toc">
              <div class="block">
                <h1>Table of Contents</h1>
                <xsl:apply-templates select="fr:tree/fr:mainmatter" mode="toc" />
              </div>
            </nav>
          </xsl:if>
        </div>
      </body>
    </html>
  </xsl:template>

  <xsl:template match="fr:tree" mode="tree-taxon-with-number">
    <xsl:param name="suffix" select="''" />
    <xsl:param name="taxon" select="fr:frontmatter/fr:taxon" />
    <xsl:param name="number" select="fr:frontmatter/fr:number" />
    <xsl:param name="fallback-number" />
    <xsl:param name="in-backmatter" select="ancestor::fr:backmatter" />

    <xsl:variable name="tree-is-root" select="not(parent::*)" />

    <xsl:variable name="explicitly-unnumbered" select="boolean(ancestor-or-self::fr:tree[@numbered='false' or @toc='false'])" />
    <xsl:variable name="implicitly-unnumbered" select="count(../fr:tree) = 1 and not(count(fr:mainmatter/fr:tree) > 1)" />

    <xsl:variable name="should-number" select="$number != '' or (not($in-backmatter) and not($tree-is-root) and not($explicitly-unnumbered)) and not($implicitly-unnumbered)" />

    <xsl:if test="$taxon != ''">
      <xsl:value-of select="$taxon" />
      <xsl:if test="$should-number or $fallback-number != ''">
        <xsl:text>&#160;</xsl:text>
      </xsl:if>
    </xsl:if>

    <xsl:choose>
      <xsl:when test="$should-number">
        <xsl:choose>
          <xsl:when test="$number != ''">
            <xsl:value-of select="$number" />
          </xsl:when>
          <xsl:otherwise>
            <xsl:number format="1.1" count="fr:tree[ancestor::fr:tree and (not(@toc='false' or @numbered='false'))]" level="multiple" />
          </xsl:otherwise>
        </xsl:choose>
      </xsl:when>
      <xsl:when test="$fallback-number != ''">
        <xsl:value-of select="$fallback-number" />
      </xsl:when>
    </xsl:choose>

    <xsl:if test="$taxon != '' or $fallback-number != '' or $should-number">
      <xsl:value-of select="$suffix" />
    </xsl:if>
  </xsl:template>

  <xsl:template match="fr:tree" mode="contextual-number">
    <xsl:param name="suffix" select="''" />
    <xsl:param name="number" select="fr:frontmatter/fr:number" />
    <xsl:param name="fallback-number" />
    <xsl:param name="in-backmatter" select="ancestor::fr:backmatter" />

    <xsl:variable name="tree-is-root" select="not(parent::*)" />

    <xsl:variable name="explicitly-unnumbered" select="boolean(ancestor-or-self::fr:tree[@numbered='false' or @toc='false'])" />
    <xsl:variable name="implicitly-unnumbered" select="count(../fr:tree) = 1 and not(count(fr:mainmatter/fr:tree) > 1)" />

    <xsl:variable name="should-number" select="$number != '' or (not($in-backmatter) and not($tree-is-root) and not($explicitly-unnumbered)) and not($implicitly-unnumbered)" />

    <xsl:choose>
      <xsl:when test="$should-number">
        <xsl:choose>
          <xsl:when test="$number != ''">
            <xsl:value-of select="$number" />
          </xsl:when>
          <xsl:otherwise>
            <xsl:number format="1.1" count="fr:tree[ancestor::fr:tree and (not(@toc='false' or @numbered='false'))]" level="multiple" />
          </xsl:otherwise>
        </xsl:choose>
      </xsl:when>
      <xsl:when test="$fallback-number != ''">
        <xsl:value-of select="$fallback-number" />
      </xsl:when>
    </xsl:choose>

    <xsl:if test="$fallback-number != '' or $should-number">
      <xsl:value-of select="$suffix" />
    </xsl:if>
  </xsl:template>

  <xsl:template match="fr:tree" mode="toc">
    <li>
      <xsl:for-each select="fr:frontmatter">
        <a class="bullet">
          <xsl:choose>
            <xsl:when test="fr:display-uri and fr:route">
              <xsl:attribute name="href">
                <xsl:value-of select="fr:route" />
              </xsl:attribute>
              <xsl:attribute name="title">
                <xsl:value-of select="fr:title/@text" />
                <xsl:text>&#160;[</xsl:text>
                <xsl:value-of select="fr:display-uri" />
                <xsl:text>]</xsl:text>
              </xsl:attribute>
            </xsl:when>
            <xsl:otherwise>
              <xsl:attribute name="href">
                <xsl:text>#</xsl:text>
                <xsl:value-of select="generate-id(..)" />
              </xsl:attribute>
              <xsl:attribute name="title">
                <xsl:value-of select="fr:title" />
              </xsl:attribute>
            </xsl:otherwise>
          </xsl:choose>
          <xsl:text>■</xsl:text>
        </a>
        <span class="link local" data-target="#{generate-id(..)}">
          <span class="taxon">
            <xsl:apply-templates select=".." mode="tree-taxon-with-number">
              <xsl:with-param name="suffix">.&#160;</xsl:with-param>
            </xsl:apply-templates>
          </span>

          <xsl:apply-templates select="fr:title" />
        </span>
      </xsl:for-each>
      <xsl:apply-templates select="fr:mainmatter" mode="toc" />
    </li>
  </xsl:template>

  <xsl:template match="fr:mainmatter" mode="toc">
    <ul class="block">
      <xsl:apply-templates select="fr:tree[not(@toc='false')]" mode="toc" />
    </ul>
  </xsl:template>

  <xsl:template match="fr:frontmatter/fr:title">
    <xsl:apply-templates />
  </xsl:template>

  <xsl:template match="fr:mainmatter">
     <xsl:apply-templates />
  </xsl:template>

  <xsl:template match="fr:display-uri[../fr:route]">
    <a class="slug" href="{../fr:route}">
      <xsl:text>[</xsl:text>
      <xsl:value-of select="." />
      <xsl:text>]</xsl:text>
    </a>
  </xsl:template>

  <xsl:template match="fr:display-uri[not(../fr:route)]">
  </xsl:template>

  <xsl:template match="fr:resource">
    <xsl:apply-templates select="fr:resource-content" />
  </xsl:template>

  <xsl:template match="fr:resource-content">
    <xsl:apply-templates />
  </xsl:template>

  <xsl:template match="fr:source-path">
    <a class="edit-button" href="{concat('vscode://file', .)}">
      <xsl:text>[edit]</xsl:text>
    </a>
  </xsl:template>

  <xsl:template match="fr:taxon">
    <xsl:value-of select="." />
  </xsl:template>

  <xsl:template match="fr:frontmatter">
    <header>
      <h1>
        <span class="taxon">
          <xsl:apply-templates select=".." mode="tree-taxon-with-number">
            <xsl:with-param name="suffix">.&#160;</xsl:with-param>
          </xsl:apply-templates>
        </span>

        <xsl:apply-templates select="fr:title" />
        <xsl:text>&#032;</xsl:text>
        <xsl:apply-templates select="fr:display-uri" />
        <xsl:text>&#032;</xsl:text>
        <xsl:apply-templates select="fr:source-path" />
      </h1>
      <div class="metadata">
        <ul>
          <xsl:apply-templates select="fr:date" />
          <xsl:if test="not(fr:meta[@name = 'author']/.='false')">
            <xsl:apply-templates select="fr:authors" />
          </xsl:if>
          <xsl:apply-templates select="fr:meta[@name='position']" />
          <xsl:apply-templates select="fr:meta[@name='institution']" />
          <xsl:apply-templates select="fr:meta[@name='venue']" />
          <xsl:apply-templates select="fr:meta[@name='source']" />
          <xsl:apply-templates select="fr:meta[@name='doi']" />
          <xsl:apply-templates select="fr:meta[@name='orcid']" />
          <xsl:apply-templates select="fr:meta[@name='external']" />
          <xsl:apply-templates select="fr:meta[@name='slides']" />
          <xsl:apply-templates select="fr:meta[@name='video']" />
        </ul>
      </div>
    </header>
  </xsl:template>

  <xsl:template match="fr:ref">
    <xsl:variable name="fallback-number">
      <xsl:text>[</xsl:text>
      <xsl:value-of select="@uri" />
      <xsl:text>]</xsl:text>
    </xsl:variable>

    <xsl:variable name="taxon">
      <xsl:choose>
        <xsl:when test="@taxon">
          <xsl:value-of select="@taxon" />
        </xsl:when>
        <xsl:otherwise>
          <xsl:text>§</xsl:text>
        </xsl:otherwise>
      </xsl:choose>
    </xsl:variable>

    <a class="link local">
      <xsl:attribute name="href">
        <xsl:choose>
          <xsl:when test="key('tree-with-uri',current()/@uri)">
            <xsl:text>#</xsl:text>
            <xsl:value-of select="generate-id(key('tree-with-uri',current()/@uri))" />
          </xsl:when>
          <xsl:otherwise>
            <xsl:value-of select="@href" />
          </xsl:otherwise>
        </xsl:choose>
      </xsl:attribute>

      <xsl:choose>
        <xsl:when test="key('tree-with-uri', current()/@uri)">
          <xsl:apply-templates select="key('tree-with-uri', current()/@uri)" mode="tree-taxon-with-number">
            <xsl:with-param name="in-backmatter" select="boolean(ancestor::fr:backmatter)" />
            <xsl:with-param name="number" select="@number" />
            <xsl:with-param name="fallback-number" select="$fallback-number" />
            <xsl:with-param name="taxon" select="$taxon" />
          </xsl:apply-templates>
        </xsl:when>
        <xsl:otherwise>
          <xsl:value-of select="$taxon" />
          <xsl:text>&#160;</xsl:text>
          <xsl:choose>
            <xsl:when test="@number">
              <xsl:value-of select="@number" />
            </xsl:when>
            <xsl:otherwise>
              <xsl:value-of select="$fallback-number" />
            </xsl:otherwise>
          </xsl:choose>
        </xsl:otherwise>
      </xsl:choose>
    </a>
  </xsl:template>

  <xsl:template match="fr:contextual-number[@uri]">
    <xsl:variable name="fallback-number">
      <xsl:text>[</xsl:text>
      <xsl:value-of select="@display-uri" />
      <xsl:text>]</xsl:text>
    </xsl:variable>

    <xsl:choose>
      <xsl:when test="key('tree-with-uri', current()/@uri)">
        <xsl:apply-templates select="key('tree-with-uri', current()/@uri)" mode="contextual-number">
          <xsl:with-param name="in-backmatter" select="boolean(ancestor::fr:backmatter)" />
          <xsl:with-param name="fallback-number" select="$fallback-number" />
        </xsl:apply-templates>
      </xsl:when>
      <xsl:otherwise>
        <xsl:value-of select="$fallback-number" />
      </xsl:otherwise>
    </xsl:choose>
  </xsl:template>

  <xsl:template match="/fr:tree[@root='true']/fr:backmatter">
  </xsl:template>

  <xsl:template match="/fr:tree[not(@root='true')]/fr:backmatter">
    <footer>
      <xsl:apply-templates />
    </footer>
  </xsl:template>

  <xsl:template match="fr:mainmatter//fr:backmatter">
  </xsl:template>

  <xsl:template match="fr:backmatter//fr:backmatter">
  </xsl:template>

  <xsl:template match="fr:tree[fr:mainmatter[*] or not(@hidden-when-empty = 'true')]">
    <section>
      <xsl:attribute name="lang">
        <xsl:choose>
          <xsl:when test="fr:frontmatter/fr:meta[@name='lang']">
            <xsl:value-of select="fr:frontmatter/fr:meta[@name='lang']" />
          </xsl:when>
          <xsl:otherwise>en</xsl:otherwise>
        </xsl:choose>
      </xsl:attribute>

      <xsl:choose>
        <xsl:when test="@show-metadata = 'false'">
          <xsl:attribute name="class">block hide-metadata</xsl:attribute>
        </xsl:when>
        <xsl:otherwise>
          <xsl:attribute name="class">block</xsl:attribute>
        </xsl:otherwise>
      </xsl:choose>
      <xsl:if test="fr:frontmatter/fr:taxon">
        <xsl:attribute name="data-taxon">
          <xsl:value-of select="fr:frontmatter/fr:taxon" />
        </xsl:attribute>
      </xsl:if>

      <xsl:choose>
        <xsl:when test="not(@show-heading='false')">
          <details id="{generate-id(.)}">
            <xsl:if test="not(@expanded = 'false')">
              <xsl:attribute name="open">open</xsl:attribute>
            </xsl:if>
            <summary>
              <xsl:apply-templates select="fr:frontmatter" />
            </summary>
            <xsl:apply-templates select="fr:mainmatter" />
            <xsl:apply-templates select="fr:frontmatter/fr:meta[@name='bibtex']" />
          </details>
        </xsl:when>
        <xsl:otherwise>
          <xsl:apply-templates select="fr:mainmatter" />
        </xsl:otherwise>
      </xsl:choose>
    </section>

    <xsl:apply-templates select="fr:backmatter" />
  </xsl:template>

  <xsl:template match="fr:tree"></xsl:template>

</xsl:stylesheet>
