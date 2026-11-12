<?xml version="1.0"?>
<!-- SPDX-License-Identifier: CC0-1.0 -->
<xsl:stylesheet version="1.0"
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:fr="http://www.jonmsterling.com/jms-005P.xml">

  <xsl:key name="tree-with-uri" match="fr:tree" use="fr:frontmatter/fr:uri" />
  <xsl:key name="tree-with-addr" match="fr:tree" use="fr:frontmatter/fr:addr" />

  <xsl:template match="/">
    <html xmlns="http://www.w3.org/1999/xhtml" data-base-url="{/fr:tree/@base-url}">
      <head>
        <meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'none'" />
        <meta name="referrer" content="strict-origin-when-cross-origin" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <link rel="stylesheet" href="{/fr:tree/@base-url}style.css" />
        <link rel="stylesheet" href="{/fr:tree/@base-url}katex.min.css" />
        <script type="module" src="{/fr:tree/@base-url}forester.js"></script>
        <script type="text/javascript" src="{/fr:tree/@base-url}assets/baux_ui.js" defer="defer"></script>
        <title>
          <xsl:value-of select="/fr:tree/fr:frontmatter/fr:title/@text" />
        </title>
      </head>
      <body>
        <xsl:variable name="current-addr" select="/fr:tree/fr:frontmatter/fr:addr" />
        <header class="masthead">
          <div class="masthead-inner">
            <div class="masthead-brand">
              <a href="{/fr:tree/@base-url}index.html" class="brand-title">baux</a>
            </div>
            <nav class="masthead-nav">
              <a href="{/fr:tree/@base-url}index.html">
                <xsl:attribute name="class">
                  <xsl:text>nav-item</xsl:text>
                  <xsl:if test="$current-addr = 'baux-0001'">
                    <xsl:text> active</xsl:text>
                  </xsl:if>
                </xsl:attribute>
                <xsl:text>Overview</xsl:text>
              </a>
              <a href="{/fr:tree/@base-url}docs.html">
                <xsl:attribute name="class">
                  <xsl:text>nav-item</xsl:text>
                  <xsl:if test="$current-addr = 'baux-0007'">
                    <xsl:text> active</xsl:text>
                  </xsl:if>
                </xsl:attribute>
                <xsl:text>API Reference</xsl:text>
              </a>
              <a href="{/fr:tree/@base-url}simulation.html">
                <xsl:attribute name="class">
                  <xsl:text>nav-item</xsl:text>
                  <xsl:if test="$current-addr = 'baux-0004'">
                    <xsl:text> active</xsl:text>
                  </xsl:if>
                </xsl:attribute>
                <xsl:text>Playground</xsl:text>
              </a>
              <a href="https://github.com/Kvutza/baux" class="nav-item external" target="_blank" rel="noopener noreferrer">
                <xsl:text>GitHub</xsl:text>
              </a>
              <a href="https://pypi.org/project/baux/" class="nav-item external" target="_blank" rel="noopener noreferrer">
                <xsl:text>PyPI</xsl:text>
              </a>
              <a href="https://crates.io/crates/baux" class="nav-item external" target="_blank" rel="noopener noreferrer">
                <xsl:text>Crates.io</xsl:text>
              </a>
            </nav>
          </div>
        </header>
        <div id="grid-wrapper">
          <article>
            <xsl:apply-templates select="fr:tree" />
          </article>
          <xsl:if test="fr:tree/fr:mainmatter/fr:tree[not(@toc='false')] and not(/fr:tree/fr:frontmatter/fr:meta[@name = 'toc']/.='false')">
            <nav id="toc">
              <div class="block">
                <h1>Contents</h1>
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
        <a class="link local" href="#{generate-id(..)}" data-target="#{generate-id(..)}">
          <span class="taxon">
            <xsl:apply-templates select=".." mode="contextual-number">
              <xsl:with-param name="suffix">.&#160;</xsl:with-param>
            </xsl:apply-templates>
          </span>
          <xsl:apply-templates select="fr:title" />
        </a>
      </xsl:for-each>
      <xsl:apply-templates select="fr:mainmatter" mode="toc" />
    </li>
  </xsl:template>

  <xsl:template match="fr:mainmatter" mode="toc">
    <xsl:if test="fr:tree[not(@toc='false')]">
      <ul class="block">
        <xsl:apply-templates select="fr:tree[not(@toc='false')]" mode="toc" />
      </ul>
    </xsl:if>
  </xsl:template>

  <xsl:template match="fr:frontmatter/fr:title">
    <xsl:apply-templates />
  </xsl:template>

  <xsl:template match="fr:mainmatter">
    <xsl:apply-templates />
  </xsl:template>

  <xsl:template match="fr:display-uri">
  </xsl:template>

  <xsl:template match="fr:resource">
    <xsl:apply-templates select="fr:resource-content" />
  </xsl:template>

  <xsl:template match="fr:resource-content">
    <xsl:apply-templates />
  </xsl:template>

  <xsl:template match="fr:source-path">
  </xsl:template>

  <xsl:template match="fr:taxon">
  </xsl:template>

  <xsl:template match="fr:frontmatter">
    <header>
      <h1>
        <xsl:variable name="sec-num">
          <xsl:apply-templates select=".." mode="contextual-number">
            <xsl:with-param name="suffix">.&#160;</xsl:with-param>
          </xsl:apply-templates>
        </xsl:variable>
        <xsl:if test="string-length($sec-num) &gt; 0">
          <span class="sec-num">
            <xsl:value-of select="$sec-num" />
          </span>
        </xsl:if>
        <span class="sec-title">
          <xsl:apply-templates select="fr:title" />
        </span>
      </h1>
    </header>
  </xsl:template>

  <xsl:template match="fr:ref">
    <xsl:variable name="target-key">
      <xsl:choose>
        <xsl:when test="@addr"><xsl:value-of select="@addr" /></xsl:when>
        <xsl:otherwise><xsl:value-of select="@uri" /></xsl:otherwise>
      </xsl:choose>
    </xsl:variable>

    <xsl:variable name="target-tree" select="key('tree-with-addr', $target-key) | key('tree-with-uri', $target-key)" />

    <xsl:variable name="fallback-number">
      <xsl:text>[</xsl:text>
      <xsl:value-of select="$target-key" />
      <xsl:text>]</xsl:text>
    </xsl:variable>

    <xsl:variable name="check-href"
      select="translate(normalize-space(@href), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ&#9;&#10;&#13; ', 'abcdefghijklmnopqrstuvwxyz')" />
    <xsl:variable name="unsafe-scheme"
      select="starts-with($check-href, 'javascript:') or starts-with($check-href, 'vbscript:') or starts-with($check-href, 'data:') or starts-with($check-href, 'blob:') or starts-with($check-href, 'file:')" />

    <a class="link local">
      <xsl:attribute name="href">
        <xsl:choose>
          <xsl:when test="$target-tree">
            <xsl:text>#</xsl:text>
            <xsl:value-of select="generate-id($target-tree[1])" />
          </xsl:when>
          <xsl:when test="$unsafe-scheme">
            <xsl:text>#</xsl:text>
          </xsl:when>
          <xsl:when test="substring(@href, string-length(@href) - 3) = '.xml'">
            <xsl:value-of select="concat(substring(@href, 1, string-length(@href) - 4), '.html')" />
          </xsl:when>
          <xsl:otherwise>
            <xsl:value-of select="@href" />
          </xsl:otherwise>
        </xsl:choose>
      </xsl:attribute>

      <xsl:text>§&#160;</xsl:text>
      <xsl:choose>
        <xsl:when test="$target-tree">
          <xsl:apply-templates select="$target-tree[1]" mode="contextual-number">
            <xsl:with-param name="in-backmatter" select="boolean(ancestor::fr:backmatter)" />
            <xsl:with-param name="number" select="@number" />
            <xsl:with-param name="fallback-number" select="$fallback-number" />
          </xsl:apply-templates>
        </xsl:when>
        <xsl:when test="@number">
          <xsl:value-of select="@number" />
        </xsl:when>
        <xsl:otherwise>
          <xsl:value-of select="$fallback-number" />
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

  <xsl:template match="/fr:tree/fr:backmatter">
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
      <xsl:attribute name="class">block</xsl:attribute>

      <xsl:choose>
        <xsl:when test="not(@show-heading='false')">
          <div class="tree-node" id="{generate-id(.)}">
            <xsl:apply-templates select="fr:frontmatter" />
            <xsl:if test="fr:frontmatter/fr:addr = 'baux-0004'">
              <div id="tokamak-cad-simulation"></div>
            </xsl:if>
            <xsl:apply-templates select="fr:mainmatter" />
          </div>
        </xsl:when>
        <xsl:otherwise>
          <xsl:if test="fr:frontmatter/fr:addr = 'baux-0004'">
            <div id="tokamak-cad-simulation"></div>
          </xsl:if>
          <xsl:apply-templates select="fr:mainmatter" />
        </xsl:otherwise>
      </xsl:choose>
    </section>
  </xsl:template>

  <xsl:template match="fr:tree"></xsl:template>

</xsl:stylesheet>
