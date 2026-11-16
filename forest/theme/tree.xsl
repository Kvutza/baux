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
        <meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'self' 'wasm-unsafe-eval'; worker-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'none'" />
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
      <xsl:variable name="current-addr" select="/fr:tree/fr:frontmatter/fr:addr" />
      <body data-page="{$current-addr}">
        <xsl:attribute name="data-docs"><xsl:choose><xsl:when test="$current-addr = 'baux-0007' or starts-with($current-addr, 'api-')">api</xsl:when><xsl:otherwise>guide</xsl:otherwise></xsl:choose></xsl:attribute>
        <a class="skip-link" href="#main-content">Skip to content</a>
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
                <xsl:text>Get started</xsl:text>
              </a>
              <a href="{/fr:tree/@base-url}docs.html">
                <xsl:attribute name="class">
                  <xsl:text>nav-item</xsl:text>
                  <xsl:if test="$current-addr = 'baux-0007' or starts-with($current-addr, 'api-')">
                    <xsl:text> active</xsl:text>
                  </xsl:if>
                </xsl:attribute>
                <xsl:text>API Reference</xsl:text>
              </a>
              <a href="{/fr:tree/@base-url}explorer.html">
                <xsl:attribute name="class">
                  <xsl:text>nav-item</xsl:text>
                  <xsl:if test="$current-addr = 'baux-0004'">
                    <xsl:text> active</xsl:text>
                  </xsl:if>
                </xsl:attribute>
                <xsl:text>Explorer</xsl:text>
              </a>
              <a href="https://github.com/Kvutza/baux" class="nav-item external" target="_blank" rel="noopener noreferrer">
                <xsl:text>GitHub</xsl:text>
              </a>
            </nav>
          </div>
        </header>
        <div class="docs-toolbar">
          <button id="docs-menu" type="button" aria-label="Open navigation" aria-expanded="false" aria-controls="docs-nav">Contents</button>
          <div class="docs-search">
            <input id="docs-search" type="search" placeholder="Search documentation" aria-label="Search documentation" aria-controls="docs-results" aria-expanded="false" autocomplete="off" spellcheck="false"/>
            <div id="docs-results" hidden="hidden"><p id="docs-search-status" role="status"></p><div id="docs-search-matches"></div></div>
          </div>
          <button id="docs-theme" type="button" aria-label="Switch color theme">◐</button>
        </div>
        <button id="docs-shade" type="button" aria-label="Close navigation" hidden="hidden"></button>
        <div id="grid-wrapper">
          <nav id="docs-nav" class="docs-sidebar" aria-label="Documentation">
            <button id="docs-close" type="button">Close contents</button>
            <a class="docs-nav-title" href="{/fr:tree/@base-url}index.html">Documentation</a>
            <ul class="docs-navigation">
              <xsl:call-template name="docs-link"><xsl:with-param name="addr" select="'baux-0001'"/><xsl:with-param name="route" select="'index.html'"/><xsl:with-param name="label" select="'Home'"/></xsl:call-template>
<xsl:call-template name="docs-link"><xsl:with-param name="addr" select="'quickstart'"/><xsl:with-param name="route" select="'quickstart.html'"/><xsl:with-param name="label" select="'Quickstart'"/></xsl:call-template>
<li class="nav-label">Guides</li>
<xsl:call-template name="docs-link"><xsl:with-param name="addr" select="'baux-0008'"/><xsl:with-param name="route" select="'baux-0008.html'"/><xsl:with-param name="label" select="'Structured parameters'"/></xsl:call-template>
<xsl:call-template name="docs-link"><xsl:with-param name="addr" select="'baux-0009'"/><xsl:with-param name="route" select="'baux-0009.html'"/><xsl:with-param name="label" select="'Regions and workers'"/></xsl:call-template>
<xsl:call-template name="docs-link"><xsl:with-param name="addr" select="'baux-0003'"/><xsl:with-param name="route" select="'baux-0003.html'"/><xsl:with-param name="label" select="'ENN predictions'"/></xsl:call-template>
<li class="nav-label">API Reference</li>
<xsl:call-template name="docs-link"><xsl:with-param name="addr" select="'baux-0007'"/><xsl:with-param name="route" select="'docs.html'"/><xsl:with-param name="label" select="'Optimizer'"/></xsl:call-template>
<xsl:call-template name="docs-link"><xsl:with-param name="addr" select="'api-pipelines'"/><xsl:with-param name="route" select="'api-pipelines.html'"/><xsl:with-param name="label" select="'Pipelines'"/></xsl:call-template>
<xsl:call-template name="docs-link"><xsl:with-param name="addr" select="'api-domains'"/><xsl:with-param name="route" select="'api-domains.html'"/><xsl:with-param name="label" select="'Parameter domains'"/></xsl:call-template>
<xsl:call-template name="docs-link"><xsl:with-param name="addr" select="'api-enn'"/><xsl:with-param name="route" select="'api-enn.html'"/><xsl:with-param name="label" select="'ENN predictions'"/></xsl:call-template>
<xsl:call-template name="docs-link"><xsl:with-param name="addr" select="'api-state'"/><xsl:with-param name="route" select="'api-state.html'"/><xsl:with-param name="label" select="'Checkpoints and keys'"/></xsl:call-template>
<xsl:call-template name="docs-link"><xsl:with-param name="addr" select="'api-configuration'"/><xsl:with-param name="route" select="'api-configuration.html'"/><xsl:with-param name="label" select="'Configuration'"/></xsl:call-template>
<xsl:call-template name="docs-link"><xsl:with-param name="addr" select="'api-objects'"/><xsl:with-param name="route" select="'api-objects.html'"/><xsl:with-param name="label" select="'In-place interfaces'"/></xsl:call-template>
<xsl:call-template name="docs-link"><xsl:with-param name="addr" select="'api-runtime'"/><xsl:with-param name="route" select="'api-runtime.html'"/><xsl:with-param name="label" select="'Runtime and integrations'"/></xsl:call-template>
<li class="nav-label">More</li>
<xsl:call-template name="docs-link"><xsl:with-param name="addr" select="'baux-0004'"/><xsl:with-param name="route" select="'explorer.html'"/><xsl:with-param name="label" select="'Explorer'"/></xsl:call-template>
<xsl:call-template name="docs-link"><xsl:with-param name="addr" select="'baux-0002'"/><xsl:with-param name="route" select="'baux-0002.html'"/><xsl:with-param name="label" select="'References'"/></xsl:call-template>
            </ul>
          </nav>
          <main id="main-content" tabindex="-1">
            <article><xsl:apply-templates select="fr:tree"/></article>
            <footer class="docs-footer"><a href="{/fr:tree/@base-url}index.html">baux</a><a href="https://pypi.org/project/baux/" rel="noopener noreferrer">PyPI</a><a href="https://crates.io/crates/baux" rel="noopener noreferrer">Crates.io</a><a href="#main-content">Back to top ↑</a></footer>
          </main>
        </div>
        <div id="docs-index" hidden="hidden">
          <xsl:call-template name="docs-index"><xsl:with-param name="addr" select="'baux-0001'"/><xsl:with-param name="route" select="'index.html'"/><xsl:with-param name="label" select="'Home'"/></xsl:call-template>
<xsl:call-template name="docs-index"><xsl:with-param name="addr" select="'quickstart'"/><xsl:with-param name="route" select="'quickstart.html'"/><xsl:with-param name="label" select="'Quickstart'"/></xsl:call-template>
<xsl:call-template name="docs-index"><xsl:with-param name="addr" select="'baux-0008'"/><xsl:with-param name="route" select="'baux-0008.html'"/><xsl:with-param name="label" select="'Structured parameters'"/></xsl:call-template>
<xsl:call-template name="docs-index"><xsl:with-param name="addr" select="'baux-0009'"/><xsl:with-param name="route" select="'baux-0009.html'"/><xsl:with-param name="label" select="'Regions and workers'"/></xsl:call-template>
<xsl:call-template name="docs-index"><xsl:with-param name="addr" select="'baux-0003'"/><xsl:with-param name="route" select="'baux-0003.html'"/><xsl:with-param name="label" select="'ENN predictions'"/></xsl:call-template>
<xsl:call-template name="docs-index"><xsl:with-param name="addr" select="'baux-0007'"/><xsl:with-param name="route" select="'docs.html'"/><xsl:with-param name="label" select="'Optimizer'"/></xsl:call-template>
<xsl:call-template name="docs-index"><xsl:with-param name="addr" select="'api-pipelines'"/><xsl:with-param name="route" select="'api-pipelines.html'"/><xsl:with-param name="label" select="'Pipelines'"/></xsl:call-template>
<xsl:call-template name="docs-index"><xsl:with-param name="addr" select="'api-domains'"/><xsl:with-param name="route" select="'api-domains.html'"/><xsl:with-param name="label" select="'Parameter domains'"/></xsl:call-template>
<xsl:call-template name="docs-index"><xsl:with-param name="addr" select="'api-enn'"/><xsl:with-param name="route" select="'api-enn.html'"/><xsl:with-param name="label" select="'ENN predictions'"/></xsl:call-template>
<xsl:call-template name="docs-index"><xsl:with-param name="addr" select="'api-state'"/><xsl:with-param name="route" select="'api-state.html'"/><xsl:with-param name="label" select="'Checkpoints and keys'"/></xsl:call-template>
<xsl:call-template name="docs-index"><xsl:with-param name="addr" select="'api-configuration'"/><xsl:with-param name="route" select="'api-configuration.html'"/><xsl:with-param name="label" select="'Configuration'"/></xsl:call-template>
<xsl:call-template name="docs-index"><xsl:with-param name="addr" select="'api-objects'"/><xsl:with-param name="route" select="'api-objects.html'"/><xsl:with-param name="label" select="'In-place interfaces'"/></xsl:call-template>
<xsl:call-template name="docs-index"><xsl:with-param name="addr" select="'api-runtime'"/><xsl:with-param name="route" select="'api-runtime.html'"/><xsl:with-param name="label" select="'Runtime and integrations'"/></xsl:call-template>
<xsl:call-template name="docs-index"><xsl:with-param name="addr" select="'baux-0004'"/><xsl:with-param name="route" select="'explorer.html'"/><xsl:with-param name="label" select="'Explorer'"/></xsl:call-template>
<xsl:call-template name="docs-index"><xsl:with-param name="addr" select="'baux-0002'"/><xsl:with-param name="route" select="'baux-0002.html'"/><xsl:with-param name="label" select="'References'"/></xsl:call-template>
        </div>
      </body>
    </html>
  </xsl:template>


  <xsl:template name="docs-link">
    <xsl:param name="addr"/><xsl:param name="route"/><xsl:param name="label"/>
    <xsl:variable name="active" select="/fr:tree/fr:frontmatter/fr:addr = $addr"/>
    <li>
      <a href="{/fr:tree/@base-url}{$route}">
        <xsl:if test="$active"><xsl:attribute name="aria-current">page</xsl:attribute></xsl:if>
        <xsl:value-of select="$label"/>
      </a>
      <xsl:if test="$active">
        <ul class="docs-page-contents">
          <xsl:for-each select="/fr:tree/fr:mainmatter//section[@data-symbol]">
            <li><a href="#{@id}"><code><xsl:value-of select="substring-after(@data-symbol, 'baux.')"/></code></a></li>
          </xsl:for-each>
          <xsl:for-each select="/fr:tree/fr:mainmatter/fr:tree">
            <li><a href="#{generate-id(.)}"><xsl:value-of select="fr:frontmatter/fr:title"/></a></li>
          </xsl:for-each>
        </ul>
      </xsl:if>
    </li>
  </xsl:template>

  <!-- Index public documentation at build time; search needs no remote service. -->
  <xsl:template name="docs-index">
    <xsl:param name="addr"/><xsl:param name="route"/><xsl:param name="label"/>
    <xsl:variable name="page" select="document(concat($addr, '.xml'))/fr:tree"/>
    <a href="{$route}" data-summary="{normalize-space($page/fr:mainmatter/fr:p[1])}">
      <xsl:attribute name="data-search"><xsl:choose>
        <xsl:when test="$page/fr:mainmatter//section[@data-symbol]"><xsl:value-of select="normalize-space($page/fr:mainmatter/fr:p)"/></xsl:when>
        <xsl:otherwise><xsl:value-of select="normalize-space($page/fr:mainmatter)"/></xsl:otherwise>
      </xsl:choose></xsl:attribute>
      <xsl:value-of select="$label"/>
    </a>
    <xsl:for-each select="$page/fr:mainmatter//section[@data-symbol]">
      <a href="{$route}#{@id}" data-summary="{normalize-space(fr:p[1])}" data-search="{normalize-space(.)}">
        <xsl:value-of select="@data-symbol"/>
      </a>
    </xsl:for-each>
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
      <xsl:variable name="level"><xsl:choose><xsl:when test="count(ancestor::fr:tree) &gt; 6">6</xsl:when><xsl:otherwise><xsl:value-of select="count(ancestor::fr:tree)"/></xsl:otherwise></xsl:choose></xsl:variable>
      <xsl:element name="h{$level}"><xsl:apply-templates select="fr:title"/></xsl:element>
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
            <xsl:apply-templates select="fr:mainmatter" />
          </div>
        </xsl:when>
        <xsl:otherwise>
          <xsl:apply-templates select="fr:mainmatter" />
        </xsl:otherwise>
      </xsl:choose>
    </section>
  </xsl:template>

  <xsl:template match="fr:tree"></xsl:template>

</xsl:stylesheet>
